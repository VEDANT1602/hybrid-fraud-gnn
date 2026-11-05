"""Review-budget ranking, with explicit nulls for undefined quantities."""
import math
import numpy as np
from sklearn.metrics import average_precision_score


def selected(rows, scores, fraction=.05):
    if not 0 <= fraction <= 1 or len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid capacity or scores")
    by_day = {}
    for i, row in enumerate(rows):
        by_day.setdefault(math.floor(row["timestamp"]), []).append(i)
    picked = []
    for indices in by_day.values():
        ordered = sorted(indices, key=lambda i: (-float(scores[i]), rows[i]["event_id"]))
        picked.extend(ordered[:math.floor(len(indices)*fraction)])
    return picked


def summarize(rows, scores, fraction=.05):
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid scores")
    y = np.array([r["label"] for r in rows], dtype=int)
    picks = selected(rows, scores, fraction)
    positives, hits = int(y.sum()), int(y[picks].sum()) if picks else 0
    return {"rows": len(rows), "positives": positives, "reviews": len(picks), "positive_reviews": hits,
            "average_precision": float(average_precision_score(y, scores)) if positives else None,
            "precision_at_capacity": hits/len(picks) if picks else None,
            "recall_at_capacity": hits/positives if positives else None,
            "actual_review_fraction": len(picks)/len(rows) if rows else None}


def paired_interval(rows, baseline, candidate, draws=200, seed=2026):
    if type(draws) is not int or draws < 1:
        raise ValueError("Need positive draws")
    if len(rows) != len(baseline) or len(rows) != len(candidate):
        raise ValueError("Length mismatch")
    days = sorted({math.floor(r["timestamp"]) for r in rows})
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(days, len(days), replace=True)
        indices = [i for day in sampled for i, r in enumerate(rows) if math.floor(r["timestamp"]) == day]
        y = np.array([rows[i]["label"] for i in indices])
        if y.sum():
            values.append(float(average_precision_score(y, np.asarray(candidate)[indices])-
                                average_precision_score(y, np.asarray(baseline)[indices])))
    return {"metric": "hybrid_minus_static_average_precision", "draws": draws, "valid_draws": len(values),
            "lower": float(np.quantile(values, .025)) if values else None,
            "upper": float(np.quantile(values, .975)) if values else None,
            "resampling": "paired test-day blocks of frozen predictions; cross-day entities remain dependent"}
