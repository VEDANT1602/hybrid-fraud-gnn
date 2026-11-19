# Fraud ranking with a fixed daily review budget

## The shared-device signal

A shared device can link coordinated accounts or ordinary customers using a public terminal. The experiment asks whether a trained graph encoder can distinguish enough additional positive cases to justify its cost over engineered graph features.

## What was knowable at scoring time

A graph built from the completed transaction table would reveal edges that an analyst could only see later. Each decision day uses a graph frozen at midnight, with the same evidence cutoff for all four tree models.

## Graph construction and model controls

1. Create three chronological cohorts with disjoint customer and device IDs. All data is original and synthetic.
2. Use only the previous seven days of activity, strictly before midnight. New devices receive an isolated default representation; within-day events become usable on the following day.
3. Train two mean-aggregation GraphSAGE-style layers on mature training outcomes. The encoder learns a function over observed attributes and neighbors, never one parameter per identity.
4. Freeze that encoder. Train identical XGBoost configurations on static features, static plus engineered graph statistics, static plus random encoder features, and static plus trained embeddings.
5. Retain validation and test predictions, the review-capacity denominator, drift slices, a paired descriptive interval and actual saved model weights.

## Training and review yield

The encoder trained for 50 fixed epochs on 2400 rows with 227 positives, on **cpu**. Its 800 trainable encoder parameters changed by L1 distance 46.774016. Training loss changed from 1.2612 to 1.1440. The tree stage used XGBoost 3.1.2 on CPU. No GPU/NVIDIA, remote inference, private dataset or paid API was used in the committed run.

The test population contains 1600 decisions and 126 synthetic positive outcomes. Daily capacity is floor(5% of that day's arrivals), using a whole-day offline ranking, not an online queue simulator.

| Model | AP | Positive reviews / reviews | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| engineered_graph_xgb | 0.1725 | 19/80 | 0.2375 | 0.1508 |
| hybrid_xgb | 0.1164 | 10/80 | 0.1250 | 0.0794 |
| random_encoder_xgb | 0.1473 | 16/80 | 0.2000 | 0.1270 |
| static_xgb | 0.1133 | 7/80 | 0.0875 | 0.0556 |

The trained hybrid changed positive reviews by **+3** versus static trees and **-9** versus engineered graph trees. The paired day-block 95% interval for hybrid-minus-static AP is **[-0.0090, 0.0166]**. It is descriptive: shared customers and rings can span days, and the interval does not represent other seeds, model fits or real populations.

### Where the result is concentrated

- original_mechanism: 800 decisions, 85 positive outcomes; static/hybrid positive reviews 7/6 at the same 40-review capacity.
- shifted_mechanism: 800 decisions, 41 positive outcomes; static/hybrid positive reviews 0/4 at the same 40-review capacity.

## When the encoder earns its cost

The highest observed available test AP belongs to **engineered_graph_xgb**. That is an exploratory observation on this fixed synthetic test, not a model-selection rule to deploy. The learned hybrid did not beat the engineered-graph control on AP; its extra training stage is not justified by this comparison.

The generator deliberately gives coordinated accounts a shared-device mechanism and a noisy profile signal. That makes this a test of implementation and a declared synthetic hypothesis, not independent evidence that a GNN improves real fraud detection. The fixed parameters and single seed are published; no seed was searched to obtain a win. Learned representations add supervised capacity and compute even though the tree configuration is identical.

The engineered-graph and untrained-encoder controls matter. If engineered statistics are as good, their simpler pipeline can be preferable. If random embeddings help too, the change is not solely evidence for representation learning. These scores are uncalibrated rankings. They must not be used to deny transactions, assign guilt or make decisions about real people.

A decision about an operational review queue would require authorized transactions scored in shadow, several training seeds and an uncertainty estimate that accounts for related cases. The relevant quantities would be additional positive reviews at equal capacity, scoring latency and retraining cost.

## Inspect and reproduce

- [Raw validation/test predictions](predictions.csv)
- [Metrics, configuration and hashes](metrics.json)
- [Data card](../DATA_CARD.md)
- [Method and primary references](../METHOD.md)

This is new independent work by Vedant Thaker, November 2026. It is not CSAA/client code, a reproduction of NVIDIA's blueprint, or validation of historical employer metrics.
