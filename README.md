# Hybrid Fraud GNN | Vedant Thaker

At the same 80-review budget, XGBoost with engineered graph statistics surfaced 19 positives on the synthetic holdout. With learned graph embeddings it surfaced 10, and with static features alone it found 7. That comparison is why the simpler graph features are the candidate worth testing next. [Per-model results](results/case-study.md) · [Held-out predictions](results/predictions.csv)

There is one clock for every model. At midnight, the graph closes over the previous seven days of customer/device observations. A transaction later that day cannot add an edge to its own features. The test cohort also uses previously unseen identities. These choices matter when ordinary customers can share a device just as coordinated accounts can.

## What the four models receive

```text
prior seven days of customer/device observations    current transaction
                  |                                         |
       time-frozen bipartite graph                  static/profile/history
                  |                                         |
       2 learned mean-SAGE layers                           |
                  |                                         |
        customer/device embeddings --------------------------+
                                    |
                          actual CPU XGBoost
                                    |
                      fixed daily review capacity
```

The learned encoder has no identity embedding table. It consumes observed attributes and neighbors, so the same trained function applies to unseen customer/device cohorts. Four tree views share one fixed configuration: **static**, **engineered graph statistics**, **random encoder**, and **trained hybrid**. The controls make a more useful question possible: is an improvement due to learned relational structure, or would a few explicit graph statistics suffice?

## Inspect the measured result

- [Generated case study](results/case-study.md), [metrics and hashes](results/metrics.json), [raw predictions](results/predictions.csv)
- [Experiment contract and primary sources](METHOD.md)
- [Original synthetic dataset and limitations](DATA_CARD.md)

![Frozen synthetic test comparison](results/comparison.svg)

The chart is generated from the same numerical artifact as the report. Results are from a single declared synthetic seed, not a real fraud benchmark. A null result or a simpler control winning stays in the report. The generator deliberately contains coordinated-device patterns; it cannot independently establish business impact.

On the frozen **1,600-decision** test cohort with **126 positives** and the same **80 reviews**:

| Tree inputs | Average precision | Positive reviews |
| --- | ---: | ---: |
| Static transaction/profile/history | 0.1133 | 7 |
| Static + engineered graph statistics | **0.1725** | **19** |
| Static + untrained encoder | 0.1473 | 16 |
| Static + trained graph encoder | 0.1164 | 10 |

The trained hybrid adds three positive reviews over static trees, but loses nine against engineered graph statistics. Its AP difference from static has a descriptive interval of **[-0.0090, 0.0166]**. This run does **not** justify adding the learned encoder. I would carry the simpler graph-statistics candidate into a representative shadow evaluation and keep the static model as the operational reference.

## Run locally without spending

Python 3.11, CPU execution. Create an isolated environment and install the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m hybrid_fraud demo --out runs/reproduction
python -m hybrid_fraud report --out runs/reproduction
python -m hybrid_fraud verify --out runs/reproduction
```

The default creates 4,800 original transactions, trains 50 encoder epochs and saves actual encoder/tree weights. No data download, API call, credential, cloud resource or provider spend is involved. Linux requirements use the smaller official CPU-only XGBoost package; the checked Windows run used full XGBoost with CPU explicitly selected. Output directories may be outside this repository; report links resolve back to its actual source documents. Test results and numerical equality may vary across hardware/library versions.

An optional `--encoder-device cuda` requires a real CUDA-enabled Torch runtime and available hardware. The checked-in run uses CPU; XGBoost stays CPU in either mode. There is no NVIDIA GPU, RAPIDS, cuGraph or production-serving result here.

## What I would decide next

Keep a strong tree as the operational reference. Compare incremental review yield against the engineered-graph control, then repeat on authorized representative data and multiple training seeds. I would add an encoder only if its measured value survives drift, latency and maintenance costs. These uncalibrated synthetic rankings do not authorize decisions about real people.

This is new independent portfolio work by **Vedant Thaker**, November 2026. It demonstrates implementation, evaluation and technical communication. It is not CSAA/client code, a reproduction of NVIDIA's blueprint, or proof of historical employer metrics. Conceptual sources are credited in [METHOD.md](METHOD.md); implementation and synthetic data are original and [MIT licensed](LICENSE).
