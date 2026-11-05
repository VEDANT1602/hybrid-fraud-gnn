"""Narrative and comparison chart from the numerical artifact."""
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())


def link(path, out):
    try:
        return os.path.relpath(path.resolve(), out.resolve()).replace(os.sep, "/")
    except ValueError:
        return path.resolve().as_uri()


def number(value):
    return "N/A" if value is None else f"{value:.4f}"


def ap_conclusions(test):
    """Undefined AP is not zero and cannot participate in a model ranking."""
    defined = {name: values["average_precision"] for name, values in test.items()
               if values["average_precision"] is not None}
    if defined:
        best = max(defined, key=defined.get)
        ranking = f"The highest observed available test AP belongs to **{best}**. That is an exploratory observation on this fixed synthetic test, not a model-selection rule to deploy."
    else:
        ranking = "Test average precision is **N/A for every model** because this test population has no positive outcomes. No best model can be named from an undefined metric."
    hybrid = test["hybrid_xgb"]["average_precision"]
    engineered = test["engineered_graph_xgb"]["average_precision"]
    if hybrid is None or engineered is None:
        comparison = "The learned-versus-engineered AP comparison is **N/A** because at least one required AP is unavailable. This comparison cannot justify another training stage."
    elif hybrid <= engineered:
        comparison = "The learned hybrid did not beat the engineered-graph control on AP; its extra training stage is not justified by this comparison."
    else:
        comparison = "The learned hybrid exceeded the engineered-graph control on AP; its extra training stage needs confirmation across independent data and training seeds."
    return ranking, comparison


def render(out):
    out = Path(out)
    metrics = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    test, train = metrics["test"], metrics["encoder_training"]
    table = ["| Model | AP | Positive reviews / reviews | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in test.items():
        table.append(f"| {name} | {number(m['average_precision'])} | {m['positive_reviews']}/{m['reviews']} | {number(m['precision_at_capacity'])} | {number(m['recall_at_capacity'])} |")
    baseline, hybrid, engineered = test["static_xgb"], test["hybrid_xgb"], test["engineered_graph_xgb"]
    interval = metrics["paired_day_interval"]
    delta = hybrid["positive_reviews"]-baseline["positive_reviews"]
    control_delta = hybrid["positive_reviews"]-engineered["positive_reviews"]
    ranking, comparison = ap_conclusions(test)
    slices = []
    for name, values in metrics["slices"].items():
        a, b = values["static_xgb"], values["hybrid_xgb"]
        slices.append(f"- {name}: {a['rows']} decisions, {a['positives']} positive outcomes; static/hybrid positive reviews {a['positive_reviews']}/{b['positive_reviews']} at the same {a['reviews']}-review capacity.")
    text = ["# Do learned graph embeddings earn another training stage?", "", "## Topic: fraud ranking under a fixed review budget", "",
            "I started with a useful tree model, then asked a narrower question: does prior customer/device context justify training an encoder as well? Shared hardware can connect coordinated accounts, but a public terminal can create the same pattern for ordinary users. A bigger graph is not automatically better evidence.", "",
            "## The ordinary shortcut", "", "Concatenating an embedding to a transaction table is easy. Building that embedding from the completed graph, including future edges, is also easy. That would answer a different question from the decision the analyst actually faced. I froze each graph at the beginning of its decision day and made a strong tree baseline the starting point.", "",
            "## My framework", "", "1. Create three chronological cohorts with disjoint customer and device IDs. All data is original and synthetic.",
            "2. Use only the previous seven days of activity, strictly before midnight. New devices receive an isolated default representation; within-day events become usable on the following day.",
            "3. Train two mean-aggregation GraphSAGE-style layers on mature training outcomes. The encoder learns a function over observed attributes and neighbors, never one parameter per identity.",
            "4. Freeze that encoder. Train identical XGBoost configurations on static features, static plus engineered graph statistics, static plus random encoder features, and static plus trained embeddings.",
            "5. Retain validation and test predictions, the review-capacity denominator, drift slices, a paired descriptive interval and actual saved model weights.", "",
            "## What actually ran", "", f"The encoder trained for {train['epochs']} fixed epochs on {train['training_rows']} rows with {train['training_positives']} positives, on **{train['device']}**. Its {train['trainable_encoder_parameters']} trainable encoder parameters changed by L1 distance {train['encoder_parameter_l1_change']:.6f}. Training loss changed from {train['losses'][0]:.4f} to {train['losses'][-1]:.4f}. The tree stage used actual XGBoost {metrics['runtime']['xgboost']} on CPU. No GPU/NVIDIA, remote inference, private dataset or paid API was used in the committed run.", "",
            f"The test population contains {baseline['rows']} decisions and {baseline['positives']} synthetic positive outcomes. Daily capacity is floor(5% of that day's arrivals), using a whole-day offline ranking, not an online queue simulator.", "", *table, "",
            f"The trained hybrid changed positive reviews by **{delta:+d}** versus static trees and **{control_delta:+d}** versus engineered graph trees. The paired day-block 95% interval for hybrid-minus-static AP is **[{number(interval['lower'])}, {number(interval['upper'])}]**. It is descriptive: shared customers and rings can span days, and the interval does not represent other seeds, model fits or real populations.", "",
            "### Where the result is concentrated", "", *slices, "",
            "## The limit changes the decision", "", ranking + " " + comparison, "",
            "The generator deliberately gives coordinated accounts a shared-device mechanism and a noisy profile signal. That makes this a test of implementation and a declared synthetic hypothesis, not independent evidence that a GNN improves real fraud detection. The fixed parameters and single seed are published; no seed was searched to obtain a win. Learned representations add supervised capacity and compute even though the tree configuration is identical.", "",
            "The engineered-graph and untrained-encoder controls matter. If engineered statistics are as good, their simpler pipeline can be preferable. If random embeddings help too, the change is not solely evidence for representation learning. These scores are uncalibrated rankings. They must not be used to deny transactions, assign guilt or make decisions about real people.", "",
            "My next decision would be a shadow evaluation on authorized representative data, with the same frozen availability rules, multiple training seeds and a group-aware uncertainty analysis. I would keep the simpler model until incremental review yield, latency, maintenance and drift behavior justify the encoder. There is no production rollout recommendation here.", "",
            "## Inspect and reproduce", "", "- [Raw validation/test predictions](predictions.csv)", "- [Metrics, configuration and hashes](metrics.json)",
            f"- [Data card]({link(ROOT/'DATA_CARD.md', out)})", f"- [Method and primary references]({link(ROOT/'METHOD.md', out)})", "",
            "This is new independent work by Vedant Thaker, November 2026. It is not CSAA/client code, a reproduction of NVIDIA's blueprint, or validation of historical employer metrics.", ""]
    (out/"case-study.md").write_bytes("\n".join(text).encode())
    maximum = max(m["average_precision"] or 0 for m in test.values())*1.2 or 1
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="385" viewBox="0 0 900 385" role="img" aria-labelledby="title desc">',
           '<title id="title">Synthetic fraud ranking: compare all controls</title>',
           '<desc id="desc">Average precision from the frozen synthetic test cohort. Values do not measure workplace outcomes.</desc>',
           '<rect width="900" height="385" fill="#f8fafc"/>',
           '<g font-family="Arial,sans-serif" fill="#18324b"><text x="25" y="34" font-size="23">Does the encoder add evidence beyond a strong tree?</text>',
           f'<text x="25" y="60" font-size="15">{baseline["rows"]} synthetic decisions; {baseline["positives"]} positive outcomes; CPU experiment</text>']
    for i, (name, m) in enumerate(test.items()):
        y = 85+i*60
        if m["average_precision"] is None:
            svg.extend([f'<text x="25" y="{y+23}" font-size="15">{html.escape(name)}</text>',
                        f'<text x="235" y="{y+23}" font-size="15">N/A (no defined average precision)</text>'])
            continue
        value = m["average_precision"]
        width = value/maximum*480
        svg.extend([f'<text x="25" y="{y+23}" font-size="15">{html.escape(name)}</text>',
                    f'<rect x="235" y="{y}" width="{width:.2f}" height="34" fill="{"#0369a1" if name == "hybrid_xgb" else "#64748b"}"/>',
                    f'<text x="{245+width:.2f}" y="{y+23}" font-size="15">{number(m["average_precision"])}</text>'])
    svg.append('<text x="25" y="362" font-size="14">Average precision; one fixed generator/training seed; no real-world capture or GPU claim.</text></g></svg>')
    (out/"comparison.svg").write_bytes("\n".join(svg).encode())
