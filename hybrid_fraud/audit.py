"""Reconcile hashes, row-level metrics and actual stored model predictions."""
import torch
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from .data import load_dataset
from .features import build_examples
from .metrics import summarize, selected
from .model import FraudNetwork, embed

ROOT = Path(__file__).resolve().parents[1]


def audit(out):
    out = Path(out).resolve()
    metrics = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    for relative, expected in metrics["source_sha256"].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Source hash mismatch: "+relative)
    for relative, expected in metrics["model_sha256"].items():
        if hashlib.sha256((out/relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Model hash mismatch: "+relative)
    packed = (out/"data/transactions.csv.gz").read_bytes()
    for name, content in [("gzip", packed), ("csv", gzip.decompress(packed)),
                          ("profiles", (out/"data/profiles.json").read_bytes())]:
        if hashlib.sha256(content).hexdigest() != metrics["dataset"][name+"_sha256"]:
            raise ValueError("Dataset hash mismatch: "+name)
    raw = (out/"predictions.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != metrics["predictions_sha256"]:
        raise ValueError("Prediction hash mismatch")
    records = list(csv.DictReader(raw.decode().splitlines()))
    events, profiles = load_dataset(out/"data")
    net = FraudNetwork().eval()
    net.load_state_dict(torch.load(out/"encoder.pt", map_location="cpu", weights_only=True))
    torch.manual_seed(metrics["config"]["seed"])
    random_net = FraudNetwork().eval()
    comparisons = 0
    for cohort in ("validation", "test"):
        batches = build_examples(events, profiles, cohort)
        rows = [e for b in batches for e in b["rows"]]
        stored = [r for r in records if r["cohort"] == cohort]
        if [r["event_id"] for r in rows] != [r["event_id"] for r in stored] or any(int(r["label"]) != e["label"] for r, e in zip(stored, rows)):
            raise ValueError("Raw decision IDs or outcomes mismatch")
        static = np.concatenate([b["static"] for b in batches])
        views = {"static_xgb": static,
                 "engineered_graph_xgb": np.concatenate([static, np.concatenate([b["graph_stats"] for b in batches])], axis=1),
                 "random_encoder_xgb": np.concatenate(embed(random_net, batches)),
                 "hybrid_xgb": np.concatenate(embed(net, batches))}
        for name, features in views.items():
            estimator = xgb.XGBClassifier(n_jobs=1)
            estimator.load_model(out/(name+".json"))
            actual = estimator.predict_proba(features)[:, 1]
            saved = np.array([float(r[name+"_score"]) for r in stored])
            if not np.allclose(actual, saved, atol=1e-6, rtol=1e-6):
                raise ValueError("Stored model does not reproduce raw scores: "+name)
            observed = summarize(rows, saved)
            if observed != metrics[cohort][name]:
                raise ValueError("Aggregate differs from raw predictions: "+name)
            picks = set(selected(rows, saved))
            if any(int(r[name+"_reviewed"]) != int(i in picks) for i, r in enumerate(stored)):
                raise ValueError("Review-selection flags differ")
            comparisons += 1
    return {"verified": True, "model_cohort_reconciliations": comparisons,
            "raw_prediction_rows": len(records), "data_rows": len(events),
            "scope": "local frozen artifacts and saved CPU predictions; no external validation"}
