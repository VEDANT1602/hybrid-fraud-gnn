import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import torch  # Must precede the other numerical imports on the tested Windows runtime.
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import xgboost
from hybrid_fraud.data import generate, validate_cohorts, available_labels, save_dataset
from hybrid_fraud.features import graph_at, build_examples, STATIC_NAMES
from hybrid_fraud.metrics import selected, summarize, paired_interval
from hybrid_fraud.model import MeanSAGE, train_encoder, embed, tensors
from hybrid_fraud.report import ap_conclusions, link


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.events, self.profiles = generate(seed=53, per_day=8, customers=24)

    def test_generator_is_deterministic_and_identifiers_do_not_name_the_latent_group(self):
        self.assertEqual((self.events, self.profiles), generate(53, 8, 24))
        self.assertEqual(len(self.events), 480)
        for event in self.events:
            self.assertRegex(event["customer_id"], r"^[0-9a-f]{16}$")
            self.assertRegex(event["device_id"], r"^[0-9a-f]{16}$")

    def test_holdout_cohorts_have_disjoint_customers_and_devices(self):
        self.assertEqual(len(validate_cohorts(self.events)), len(self.events))
        altered = copy.deepcopy(self.events)
        next(e for e in altered if e["cohort"] == "test")["device_id"] = altered[0]["device_id"]
        with self.assertRaises(ValueError): validate_cohorts(altered)

    def test_duplicate_retries_are_idempotent_and_conflicts_rejected(self):
        self.assertEqual(len(validate_cohorts(self.events+[dict(self.events[0])])), len(self.events))
        wrong = {**self.events[0], "label": 1-self.events[0]["label"]}
        with self.assertRaises(ValueError): validate_cohorts(self.events+[wrong])

    def test_immature_training_labels_fail_before_fitting(self):
        with self.assertRaises(ValueError): available_labels(self.events, "train", 29)
        self.assertEqual(len(available_labels(self.events, "train", 35)), 240)

    def test_nonfinite_decision_or_availability_cannot_bypass_time_guards(self):
        latest = max((e for e in self.events if e["cohort"] == "train"), key=lambda e: e["timestamp"])
        cutoff = latest["timestamp"]
        with self.assertRaises(ValueError):
            available_labels([latest], "train", cutoff)
        for field in ("timestamp", "label_available_at"):
            for value in (float("nan"), float("inf"), float("-inf"), True, "30"):
                changed = {**latest, field: value}
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError): validate_cohorts([changed])
                    with self.assertRaises(ValueError): available_labels([changed], "train", cutoff)

    def test_label_asof_boundary_is_finite_and_exact_arrival_is_available(self):
        row = self.events[0]
        arrival = row["label_available_at"]
        self.assertEqual(available_labels([row], "train", arrival), [row["label"]])
        with self.assertRaises(ValueError): available_labels([row], "train", arrival-1e-6)
        for cutoff in (float("nan"), float("inf"), float("-inf"), True, "35"):
            with self.subTest(as_of=cutoff), self.assertRaises(ValueError):
                available_labels([row], "train", cutoff)

    def test_future_and_same_day_edges_cannot_change_frozen_graph(self):
        before = graph_at(self.events, self.profiles, "train", 10)
        altered = copy.deepcopy(self.events)
        for row in altered:
            if row["timestamp"] >= 10:
                row["device_id"] = "unseen-future-device"
                row["log_amount"] = 99
        after = graph_at(altered, self.profiles, "train", 10)
        self.assertEqual(before["index"], after["index"])
        np.testing.assert_array_equal(before["x"], after["x"])
        np.testing.assert_array_equal(before["adj"], after["adj"])

    def test_window_uses_inclusive_lower_exclusive_upper_bound(self):
        base = self.events[0]
        rows = [{**base, "event_id": str(i), "timestamp": ts} for i, ts in enumerate((2.999, 3., 9.999, 10.))]
        graph = graph_at(rows, self.profiles, "train", 10)
        self.assertEqual(graph["eligible_ids"], ["1", "2"])

    def test_outcomes_and_label_arrival_never_enter_graph_or_static_features(self):
        original = build_examples(self.events, self.profiles, "train")
        altered = [{**e, "label": 1-e["label"], "label_available_at": e["timestamp"]+100} for e in self.events]
        changed = build_examples(altered, self.profiles, "train")
        for a, b in zip(original, changed):
            for key in ("x", "adj", "static", "graph_stats"):
                np.testing.assert_array_equal(a[key], b[key])

    def test_gzip_hashes_and_data_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            manifest = save_dataset(folder, self.events, self.profiles)
            packed = (folder/"transactions.csv.gz").read_bytes()
            raw = gzip.decompress(packed)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["csv_sha256"])
            self.assertEqual(len(list(csv.DictReader(io.StringIO(raw.decode())))), len(self.events))
            save_dataset(folder, self.events, self.profiles)
            self.assertEqual(packed, (folder/"transactions.csv.gz").read_bytes())


class ModelContracts(unittest.TestCase):
    def test_encoder_is_equivariant_to_node_order_without_identity_parameters(self):
        torch.manual_seed(11)
        model = MeanSAGE().eval()
        x = torch.randn(5, 8)
        adj = torch.tensor([[0,.5,.5,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0.]])
        perm = torch.tensor([3, 1, 4, 0, 2])
        with torch.inference_mode():
            original = model(x, adj)
            changed = model(x[perm], adj[perm][:, perm])
        self.assertTrue(torch.allclose(changed, original[perm], atol=1e-6))
        self.assertFalse(any(isinstance(module, torch.nn.Embedding) for module in model.modules()))

    def test_training_changes_actual_encoder_parameters_and_handles_unseen_graph(self):
        events, profiles = generate(53, 8, 24)
        train = build_examples(events, profiles, "train")
        net, result = train_encoder(train, epochs=2)
        self.assertGreater(result["encoder_parameter_l1_change"], 0)
        outputs = embed(net, build_examples(events, profiles, "test"))
        self.assertEqual(sum(len(x) for x in outputs), 160)
        self.assertTrue(all(np.isfinite(x).all() and x.shape[1] == len(STATIC_NAMES)+16 for x in outputs))
        with tempfile.TemporaryDirectory() as temp:
            from hybrid_fraud.model import FraudNetwork
            path = Path(temp)/"encoder.pt"
            torch.save(net.state_dict(), path)
            restored = FraudNetwork().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            np.testing.assert_allclose(embed(net, train)[0], embed(restored, train)[0], atol=1e-7)

    def test_message_passing_changes_representation_when_observed_neighbors_change(self):
        torch.manual_seed(53)
        model = MeanSAGE().eval()
        x = torch.eye(8)[:3]
        empty = torch.zeros(3, 3)
        linked = torch.tensor([[0., 1, 0], [1, 0, 0], [0, 0, 0]])
        with torch.inference_mode():
            self.assertFalse(torch.allclose(model(x, empty)[0], model(x, linked)[0]))

    def test_actual_xgboost_probability_and_saved_model_round_trip(self):
        x = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = xgboost.XGBClassifier(n_estimators=3, min_child_weight=0, n_jobs=1, device="cpu", tree_method="hist")
        model.fit(x, y)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"model.json"; model.save_model(path)
            loaded = xgboost.XGBClassifier(); loaded.load_model(path)
            np.testing.assert_allclose(model.predict_proba(x), loaded.predict_proba(x))


class MetricContracts(unittest.TestCase):
    def rows(self):
        return [{"event_id": key, "timestamp": day, "label": label} for key, day, label in
                [("c", 0.1, 1), ("a", .2, 0), ("b", .3, 1), ("d", 1.2, 0)]]

    def test_budget_is_daily_floor_and_ties_use_id(self):
        rows = self.rows()
        self.assertEqual(selected(rows, np.ones(4), .5), [1])
        self.assertEqual(selected(rows, np.array([4, 1, 2, 3]), .5), [0])

    def test_average_precision_and_capacity_match_hand_calculation(self):
        result = summarize(self.rows(), np.array([.9, .8, .7, .6]), .5)
        self.assertAlmostEqual(result["average_precision"], (1+2/3)/2)
        self.assertEqual((result["reviews"], result["positive_reviews"]), (1, 1))
        self.assertEqual(result["precision_at_capacity"], 1)
        self.assertEqual(result["recall_at_capacity"], .5)

    def test_zero_positive_and_zero_capacity_are_explicit_nulls(self):
        result = summarize([{**r, "label": 0} for r in self.rows()], np.arange(4), .01)
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["precision_at_capacity"])
        self.assertIsNone(result["recall_at_capacity"])

    def test_identical_predictions_have_zero_paired_interval(self):
        interval = paired_interval(self.rows(), np.arange(4), np.arange(4), draws=30)
        self.assertEqual((interval["lower"], interval["upper"]), (0, 0))

    def test_all_negative_bootstrap_is_explicitly_unavailable(self):
        rows = [{**r, "label": 0} for r in self.rows()]
        interval = paired_interval(rows, np.arange(4), np.arange(4)[::-1], draws=30)
        self.assertEqual(interval["valid_draws"], 0)
        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError): selected(self.rows(), np.array([0, 1, np.nan, 2]))

    def test_external_report_link_resolves(self):
        card = Path(__file__).resolve().parents[1]/"DATA_CARD.md"
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual((Path(temp)/link(card, Path(temp))).resolve(), card.resolve())


class ReportContracts(unittest.TestCase):
    def test_some_or_all_undefined_ap_cannot_be_ranked_as_zero(self):
        names = ("static_xgb", "engineered_graph_xgb", "random_encoder_xgb", "hybrid_xgb")
        empty = {name: {"average_precision": None} for name in names}
        ranking, comparison = ap_conclusions(empty)
        self.assertIn("No best model", ranking)
        self.assertIn("N/A", comparison)
        for absent in ("hybrid_xgb", "engineered_graph_xgb"):
            values = {name: {"average_precision": 0.0} for name in names}
            values[absent]["average_precision"] = None
            ranking, comparison = ap_conclusions(values)
            self.assertIn("N/A", comparison)
            self.assertNotIn("**"+absent+"**", ranking)
        # An unavailable static control must not block a valid comparison of the other two.
        values = {name: {"average_precision": .1} for name in names}
        values["static_xgb"]["average_precision"] = None
        values["hybrid_xgb"]["average_precision"] = .2
        ranking, comparison = ap_conclusions(values)
        self.assertIn("**hybrid_xgb**", ranking)
        self.assertIn("exceeded", comparison)

    def test_valid_small_run_without_test_positives_finishes_and_audits(self):
        from hybrid_fraud.experiment import run
        from hybrid_fraud.audit import audit
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = run(out, seed=0, epochs=1, per_day=1)
            self.assertEqual(result["encoder_training"]["training_positives"], 1)
            self.assertTrue(all(m["positives"] == 0 and m["average_precision"] is None
                                for m in result["test"].values()))
            self.assertEqual(result["paired_day_interval"]["valid_draws"], 0)
            report = (out/"case-study.md").read_text(encoding="utf-8")
            self.assertIn("No best model", report)
            self.assertIn("[N/A, N/A]", report)
            chart = (out/"comparison.svg").read_text(encoding="utf-8")
            self.assertEqual(chart.count("N/A (no defined average precision)"), 4)
            self.assertEqual(audit(out)["model_cohort_reconciliations"], 8)


if __name__ == "__main__":
    unittest.main()
