"""Daily frozen graphs: only events in [day-7, day) are eligible."""
import math
import numpy as np
from .data import COHORTS, validate_cohorts

STATIC_NAMES = ["log_amount", "country_changed", "mobile", "profile_score", "log_account_age",
                "sin_time", "cos_time", "customer_prior_events", "customer_prior_mean_amount"]
NODE_NAMES = ["is_customer", "is_device", "profile_score", "log_account_age", "log_degree",
              "log_events", "mean_log_amount", "mean_country_changed"]
GRAPH_NAMES = ["customer_degree", "device_degree", "device_events", "device_mean_profile",
               "customer_mean_device_degree", "device_mean_amount"]


def graph_at(events, profiles, cohort, day, window=7):
    customers = sorted(k for k, v in profiles.items() if v["cohort"] == cohort and v["created_at"] <= day)
    eligible = [e for e in events if e["cohort"] == cohort and day - window <= e["timestamp"] < day]
    # Only historical device IDs are materialized. A new decision device has an isolated default node.
    devices = sorted({e["device_id"] for e in eligible})
    keys = [("customer", k) for k in customers] + [("device", k) for k in devices]
    index = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    adjacent = np.zeros((n, n), dtype=np.float32)
    features = np.zeros((n, len(NODE_NAMES)), dtype=np.float32)
    buckets = {key: [] for key in keys}
    for e in eligible:
        c, d = index[("customer", e["customer_id"])], index[("device", e["device_id"])]
        adjacent[c, d] = adjacent[d, c] = 1.
        buckets[("customer", e["customer_id"])].append(e)
        buckets[("device", e["device_id"])].append(e)
    degree = adjacent.sum(1)
    for key, i in index.items():
        kind, identity = key
        rows = buckets[key]
        profile = profiles[identity] if kind == "customer" else None
        features[i] = [float(kind == "customer"), float(kind == "device"),
                       profile["profile_score"] if profile else 0.,
                       math.log1p(profile["account_age"]) / 7 if profile else 0.,
                       math.log1p(float(degree[i])) / 5, math.log1p(len(rows)) / 7,
                       sum(e["log_amount"] for e in rows) / len(rows) / 8 if rows else 0.,
                       sum(e["country_changed"] for e in rows) / len(rows) if rows else 0.]
    normalized = adjacent / np.maximum(degree[:, None], 1)
    return {"x": features, "adj": normalized, "index": index, "buckets": buckets,
            "degree": degree, "eligible_ids": [e["event_id"] for e in eligible]}


def build_examples(events, profiles, cohort):
    events = validate_cohorts(events)
    rows = [e for e in events if e["cohort"] == cohort]
    batches = []
    for day in sorted({math.floor(e["timestamp"]) for e in rows}):
        graph = graph_at(events, profiles, cohort, day)
        daily = [e for e in rows if math.floor(e["timestamp"]) == day]
        # One shared isolated device node has exactly the observed default features.
        x = np.concatenate([graph["x"], np.array([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)])
        adj = np.zeros((len(x), len(x)), dtype=np.float32)
        adj[:-1, :-1] = graph["adj"]
        static, simple_graph, customer_indices, device_indices = [], [], [], []
        for e in daily:
            c = graph["index"][("customer", e["customer_id"])]
            d = graph["index"].get(("device", e["device_id"]), len(x)-1)
            cr = graph["buckets"][("customer", e["customer_id"])]
            dr = graph["buckets"].get(("device", e["device_id"]), [])
            hour = e["timestamp"] % 1
            static.append([e["log_amount"]/8, e["country_changed"], e["mobile"], e["profile_score"],
                           math.log1p(e["account_age"])/7, math.sin(2*math.pi*hour), math.cos(2*math.pi*hour),
                           math.log1p(len(cr))/7, sum(r["log_amount"] for r in cr)/len(cr)/8 if cr else 0.])
            neighbor_cs = {r["customer_id"] for r in dr}
            neighbor_ds = {r["device_id"] for r in cr}
            simple_graph.append([math.log1p(float(graph["degree"][c])),
                                 math.log1p(len(neighbor_cs)), math.log1p(len(dr)),
                                 sum(profiles[k]["profile_score"] for k in neighbor_cs)/len(neighbor_cs) if neighbor_cs else 0.,
                                 sum(float(graph["degree"][graph["index"][("device", k)]]) for k in neighbor_ds)/len(neighbor_ds) if neighbor_ds else 0.,
                                 sum(r["log_amount"] for r in dr)/len(dr) if dr else 0.])
            customer_indices.append(c); device_indices.append(d)
        batches.append({"day": day, "rows": daily, "x": x, "adj": adj,
                        "static": np.asarray(static, dtype=np.float32),
                        "graph_stats": np.asarray(simple_graph, dtype=np.float32),
                        "customers": np.array(customer_indices), "devices": np.array(device_indices)})
    return batches
