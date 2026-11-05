"""Deterministic, fictional customer/device activity with delayed outcomes."""
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from numbers import Real
import random


COHORTS = {"train": (0, 30), "validation": (35, 45), "test": (50, 70)}
FIELDS = ["event_id", "cohort", "timestamp", "customer_id", "device_id", "log_amount",
          "country_changed", "mobile", "profile_score", "account_age", "label", "label_available_at"]


def finite_time(value):
    """Accept finite real times, rejecting booleans, strings and unknown arrival."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def validate_times(event):
    if not finite_time(event["timestamp"]) or not finite_time(event["label_available_at"]):
        raise ValueError("Decision and label availability times must be finite real numbers")
    if event["label_available_at"] <= event["timestamp"]:
        raise ValueError("Outcome must arrive after decision")


def generate(seed=53, per_day=80, customers=120):
    if type(seed) is not int or type(per_day) is not int or per_day < 1 or customers < 12:
        raise ValueError("Invalid generator configuration")
    rng = random.Random(seed)
    events, profiles = [], {}
    for cohort, (start, end) in COHORTS.items():
        ids = [hashlib.sha256(f"{seed}/{cohort}/customer/{i}".encode()).hexdigest()[:16] for i in range(customers)]
        personal = [hashlib.sha256(f"{seed}/{cohort}/device/{i}".encode()).hexdigest()[:16] for i in range(customers)]
        shared = [hashlib.sha256(f"{seed}/{cohort}/shared/{i}".encode()).hexdigest()[:16] for i in range(12)]
        ring_count = customers // 4
        for i, customer in enumerate(ids):
            # A noisy profile signal is known at cohort start. Membership is never exported.
            profiles[customer] = {"cohort": cohort, "created_at": float(start),
                                  "profile_score": round(min(.99, max(.01, rng.gauss(.57 if i < ring_count else .39, .22))), 6),
                                  "account_age": round(rng.uniform(2, 730), 6)}
        for day in range(start, end):
            for j in range(per_day):
                i = rng.randrange(customers)
                coordinated = i < ring_count
                shifted = cohort == "test" and day >= 60
                if coordinated and rng.random() < (.45 if shifted else .86):
                    device = shared[(i // 5) % 6]
                elif rng.random() < (.35 if shifted else .20):
                    device = shared[6 + rng.randrange(6)]
                else:
                    device = personal[i]
                country, mobile = int(rng.random() < .12), int(rng.random() < .70)
                amount = round(min(8., max(1., rng.gauss(4.1 + .20 * coordinated, 1.1))), 6)
                # No labels are attached to graph edges. The mechanism deliberately gives context value.
                p = .018 + .025 * country + .022 * (amount > 5.3) + .035 * profiles[ids[i]]["profile_score"]
                p += (.11 if shifted else .24) * coordinated
                label = int(rng.random() < p)
                timestamp = day + (j + 1) / (per_day + 1)
                events.append({"event_id": f"e-{cohort}-{day:02}-{j:03}", "cohort": cohort,
                               "timestamp": timestamp, "customer_id": ids[i], "device_id": device,
                               "log_amount": amount, "country_changed": country, "mobile": mobile,
                               "profile_score": profiles[ids[i]]["profile_score"],
                               "account_age": profiles[ids[i]]["account_age"], "label": label,
                               "label_available_at": timestamp + rng.randint(1, 4)})
    return events, profiles


def validate_cohorts(events):
    events = list(events)
    # Validate before ordering or comparisons: NaN is neither earlier nor later.
    for event in events:
        validate_times(event)
    seen = {"customer_id": {}, "device_id": {}}
    ids = {}
    result = []
    for event in sorted(events, key=lambda row: (row["timestamp"], row["event_id"])):
        key = event["event_id"]
        if key in ids:
            if ids[key] != event:
                raise ValueError("Conflicting duplicate event")
            continue
        ids[key] = event
        split = event["cohort"]
        if split not in COHORTS or not COHORTS[split][0] <= event["timestamp"] < COHORTS[split][1]:
            raise ValueError("Event outside declared cohort interval")
        if type(event["label"]) is not int or event["label"] not in (0, 1):
            raise ValueError("Invalid outcome")
        for field in seen:
            value = event[field]
            if value in seen[field] and seen[field][value] != split:
                raise ValueError("Entity crosses holdout boundary")
            seen[field][value] = split
        result.append(event)
    return result


def available_labels(events, cohort, as_of):
    if not finite_time(as_of):
        raise ValueError("Label as-of boundary must be a finite real number")
    if cohort not in COHORTS:
        raise ValueError("Unknown cohort")
    rows = [e for e in events if e["cohort"] == cohort]
    for event in rows:
        validate_times(event)
    if any(e["label_available_at"] > as_of for e in rows):
        raise ValueError("Immature labels at training/audit boundary")
    return [e["label"] for e in rows]


def save_dataset(directory, events, profiles):
    directory.mkdir(parents=True, exist_ok=True)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader(); writer.writerows(events)
    raw = text.getvalue().encode()
    packed = gzip.compress(raw, mtime=0)
    (directory / "transactions.csv.gz").write_bytes(packed)
    profile_bytes = (json.dumps(profiles, sort_keys=True, indent=2) + "\n").encode()
    (directory / "profiles.json").write_bytes(profile_bytes)
    return {"rows": len(events), "csv_sha256": hashlib.sha256(raw).hexdigest(),
            "gzip_sha256": hashlib.sha256(packed).hexdigest(),
            "profiles_sha256": hashlib.sha256(profile_bytes).hexdigest()}


def load_dataset(directory):
    raw = gzip.decompress((directory/"transactions.csv.gz").read_bytes())
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    for row in rows:
        for field in ("timestamp", "log_amount", "profile_score", "account_age", "label_available_at"):
            row[field] = float(row[field])
        for field in ("country_changed", "mobile", "label"):
            row[field] = int(row[field])
    return validate_cohorts(rows), json.loads((directory/"profiles.json").read_text(encoding="utf-8"))
