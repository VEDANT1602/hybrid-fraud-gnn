"""Run the fixed CPU tree comparison with a learned inductive encoder."""
import torch  # Import first on the tested Windows numerical runtime.
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import numpy as np
import sklearn
import xgboost as xgb
from .data import COHORTS, generate, validate_cohorts, available_labels, save_dataset
from .features import build_examples, STATIC_NAMES, GRAPH_NAMES
from .metrics import summarize, selected, paired_interval
from .model import FraudNetwork, train_encoder, embed
from .report import write_json, render

ROOT = Path(__file__).resolve().parents[1]
TREE = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
        "n_estimators": 180, "max_depth": 3, "learning_rate": .05, "min_child_weight": 8,
        "subsample": 1., "colsample_bytree": 1., "reg_lambda": 4., "n_jobs": 1, "random_state": 53}


def run(out, seed=53, epochs=50, per_day=80, encoder_device="cpu"):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    events, profiles = generate(seed, per_day)
    events = validate_cohorts(events)
    dataset = save_dataset(out/"data", events, profiles)
    training_labels = np.asarray(available_labels(events, "train", 35))
    batches = {c: build_examples(events, profiles, c) for c in COHORTS}
    net, training = train_encoder(batches["train"], seed=seed, epochs=epochs, device=encoder_device)
    torch.save(net.state_dict(), out/"encoder.pt")
    # Random features are a diagnostic for extra nonlinear feature capacity, not a trained GNN.
    torch.manual_seed(seed)
    random_net = FraudNetwork().to(encoder_device).eval()
    views = {}
    for cohort, bs in batches.items():
        static = np.concatenate([b["static"] for b in bs])
        stats = np.concatenate([b["graph_stats"] for b in bs])
        views[cohort] = {"static_xgb": static, "engineered_graph_xgb": np.concatenate([static, stats], axis=1),
                         "random_encoder_xgb": np.concatenate(embed(random_net, bs, encoder_device)),
                         "hybrid_xgb": np.concatenate(embed(net, bs, encoder_device))}
    predictions = {c: {} for c in ("validation", "test")}
    timings, model_hashes = {}, {}
    for name in views["train"]:
        started = time.perf_counter()
        estimator = xgb.XGBClassifier(**{**TREE, "random_state": seed})
        estimator.fit(views["train"][name], training_labels)
        timings[name] = time.perf_counter()-started
        path = out/(name+".json")
        estimator.save_model(path)
        model_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cohort in predictions:
            predictions[cohort][name] = estimator.predict_proba(views[cohort][name])[:, 1]
    # Held-out outcomes are consulted only after all model parameters and scores are frozen.
    available_labels(events, "validation", 50)
    available_labels(events, "test", 75)
    all_rows = {c: [e for b in bs for e in b["rows"]] for c, bs in batches.items()}
    metrics = {"schema_version": 1, "scope": "original_synthetic_inductive_GNN_plus_actual_XGBoost; no workplace validation",
               "config": {"seed": seed, "epochs": epochs, "per_day": per_day, "graph_window_days": 7,
                          "graph_cutoff": "midnight_before_decision_day", "review_fraction": .05,
                          "tree_parameters": {**TREE, "random_state": seed}, "model_selection": "fixed config; no validation/test tuning",
                          "train_label_asof": 35, "validation_label_asof": 50, "test_label_asof": 75,
                          "encoder_device": encoder_device, "tree_device": "cpu"},
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "xgboost": xgb.__version__, "cuda_available": torch.cuda.is_available(),
                           "machine": platform.machine(), "processor": platform.processor()},
               "dataset": dataset, "encoder_training": training, "tree_fit_seconds": timings,
               "features": {"static": STATIC_NAMES, "engineered_graph": GRAPH_NAMES, "hybrid_embedding_dimensions": 16},
               "validation": {n: summarize(all_rows["validation"], p) for n, p in predictions["validation"].items()},
               "test": {n: summarize(all_rows["test"], p) for n, p in predictions["test"].items()},
               "slices": {}, "paired_day_interval": paired_interval(all_rows["test"], predictions["test"]["static_xgb"], predictions["test"]["hybrid_xgb"]),
               "model_sha256": {**model_hashes, "encoder.pt": hashlib.sha256((out/"encoder.pt").read_bytes()).hexdigest()},
               "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for folder in (ROOT/"hybrid_fraud", ROOT/"tests") for p in sorted(folder.glob("*.py"))}}
    for slice_name, predicate in [("original_mechanism", lambda e: e["timestamp"] < 60),
                                  ("shifted_mechanism", lambda e: e["timestamp"] >= 60)]:
        indices = [i for i, e in enumerate(all_rows["test"]) if predicate(e)]
        metrics["slices"][slice_name] = {n: summarize([all_rows["test"][i] for i in indices], p[indices]) for n, p in predictions["test"].items()}
    text = io.StringIO(newline="")
    names = list(views["train"])
    fields = ["event_id", "cohort", "timestamp", "label"] + [n+suffix for n in names for suffix in ("_score", "_reviewed")]
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n"); writer.writeheader()
    for cohort in predictions:
        picks = {n: set(selected(all_rows[cohort], p)) for n, p in predictions[cohort].items()}
        for i, row in enumerate(all_rows[cohort]):
            record = {k: row[k] for k in ("event_id", "cohort", "timestamp", "label")}
            for name in names:
                record[name+"_score"] = float(predictions[cohort][name][i]); record[name+"_reviewed"] = int(i in picks[name])
            writer.writerow(record)
    (out/"predictions.csv").write_bytes(text.getvalue().encode())
    metrics["predictions_sha256"] = hashlib.sha256((out/"predictions.csv").read_bytes()).hexdigest()
    metrics["total_seconds"] = time.perf_counter()-began
    write_json(out/"metrics.json", metrics); render(out)
    return metrics
