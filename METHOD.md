# An inductive encoder, then a tree: the experiment contract

The starting question is whether learned graph context improves a useful transaction tree at the same daily review capacity. This implementation is original. The design follows a two-stage idea, not any copied training code or published benchmark score.

## The observed boundary

Every cohort is a distinct customer/device population in a later time interval. Each day has one graph frozen before the day's decisions. A seven-day bipartite graph connects known customer profiles to observed devices; normalized adjacency implements an exact full-neighbor mean. Two learned layers concatenate a node's own representation with that mean, apply a linear transform/ReLU and L2 normalization. This is a small GraphSAGE-style mean encoder, with no per-ID embedding table and no neighbor-sampling implementation. Its parameters can encode previously unseen graphs.

The node features are type, known profile score/age, prior degree/activity, prior mean amount and prior country-change share. The current transaction's country, amount, mobile flag, profile, time of day and own prior-history summaries form a nine-feature static baseline. The classifier sees current customer and device embeddings separately, so an unseen current link is not inserted into the historical graph.

## Actual training and controls

The GNN and a temporary transaction head receive mature training labels only. Fifty fixed full-batch epochs use weighted binary cross-entropy, Adam, gradient clipping and a fixed seed; there is no test/validation epoch selection. The temporary head is discarded after training. Frozen eight-dimensional customer/device embeddings are concatenated to the same nine static features, and a real XGBoost classifier is fitted to training outcomes. Using supervised representation learning on training rows is intentional; no held-out outcome is used to learn representations or fit trees.

All four tree views use 180 histogram trees, depth 3, learning rate 0.05, minimum child weight 8 and L2 penalty 4, one CPU thread, identical rows and seed. Parameters are fixed before the first run, with no grid/early stopping. The static tree is a substantive model with profile and behavioral history. The engineered graph control adds six explicit degree/activity/neighbor-profile statistics. An untrained encoder control has the same architecture/seed, without optimization. Extra learned capacity and encoder training compute are real costs; identical trees do not imply equal end-to-end complexity.

Training cohorts and labels are sealed before fit. Validation is diagnostic, never used to select these committed parameters; the final model is not refitted on validation. Test labels are opened only to evaluate a frozen pipeline. For a real study, repeat this contract across generator/training seeds and an authorized data source; do not choose the most favorable test seed.

## Metrics and inference limits

Average precision weights precision at each positive-recall increment; undefined all-negative populations are null. Daily reviews are floor(5% of that day's decisions), with descending scores and lexical event-ID ties. Precision uses actual reviews as denominator; recall uses all positive outcomes. Capacity is fixed across models. A descriptive paired bootstrap resamples test-day blocks of fixed predictions. Customers/rings span days, so this interval does not account for all dependence or retraining uncertainty. Drift slices are declared before evaluation at test day 60.

The committed run uses CPU Torch and CPU XGBoost. `--encoder-device cuda` only runs if actual Torch CUDA is available; trees remain CPU. CUDA execution and performance are not established by this CPU run. It does not use NVIDIA cuGraph, RAPIDS, WholeGraph, DGL, PyG, Triton or NVIDIA's training container. Dense graph matrices and full-neighbor aggregation are deliberately small; no distributed-scaling, serving latency or memory-efficiency claim is made.

## Primary sources and conceptual credit

- [NVIDIA's official financial-fraud overview](https://docs.nvidia.com/nim/financial-fraud-training/latest/overview/overview.html) and [technical blueprint article](https://developer.nvidia.com/blog/supercharging-fraud-detection-in-financial-services-with-graph-neural-networks/) describe GNN embeddings followed by XGBoost. This repository implements a smaller, independent CPU experiment and does not reproduce their code, dataset, infrastructure or scores.
- [Hamilton, Ying and Leskovec: Inductive Representation Learning on Large Graphs, 2017](https://arxiv.org/abs/1706.02216) motivates a shared attribute/neighborhood aggregation function for previously unseen nodes. The exact mean layers here are original PyTorch code.
- [XGBoost parameters](https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html) and [Python API](https://xgboost.readthedocs.io/en/release_3.1.0/python/python_api.html) support the installed 3.1.2 interface and explicit CPU histogram tree configuration. The 3.1 documentation branch currently identifies 3.1.3; the installed version is recorded separately. The [official minimal-installation guide](https://xgboost.readthedocs.io/en/release_3.1.0/install.html#minimal-installation-cpu-only) and [3.1.2 CPU wheel listing](https://pypi.org/project/xgboost-cpu/3.1.2/) document the smaller CPU-only package used by Linux CI; the checked Windows run used the full package with CPU explicitly selected.
- The user's [LinkedIn post reference from Andres Vourakis](https://www.linkedin.com/posts/andresvourakis_xgboost-and-similar-tree-based-models-are-share-7523776256539029505-auas/) motivated the question of why trees remain useful alongside graph/deep learning. The post's employer anecdotes or metrics are not evidence generated by this project.

All links were consulted or supplied as context in November 2026. No affiliation with NVIDIA, XGBoost, the cited authors or a former employer is implied.
