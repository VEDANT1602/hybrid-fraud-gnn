"""Two learned mean-aggregation layers, with no ID embeddings or graph library."""
import copy
import time
import torch
from .features import NODE_NAMES, STATIC_NAMES


class MeanSAGE(torch.nn.Module):
    def __init__(self, inputs=len(NODE_NAMES), hidden=24, dimensions=8):
        super().__init__()
        self.first = torch.nn.Linear(2 * inputs, hidden)
        self.second = torch.nn.Linear(2 * hidden, dimensions)

    def forward(self, x, adjacency):
        h = torch.relu(self.first(torch.cat([x, adjacency @ x], dim=-1)))
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
        h = torch.relu(self.second(torch.cat([h, adjacency @ h], dim=-1)))
        return torch.nn.functional.normalize(h, p=2, dim=-1)


class FraudNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MeanSAGE()
        self.head = torch.nn.Sequential(torch.nn.Linear(len(STATIC_NAMES)+16, 16), torch.nn.ReLU(),
                                        torch.nn.Linear(16, 1))

    def transaction_features(self, batch):
        z = self.encoder(batch["x"], batch["adj"])
        return torch.cat([batch["static"], z[batch["customers"]], z[batch["devices"]]], dim=-1)

    def forward(self, batch):
        return self.head(self.transaction_features(batch)).squeeze(-1)


def tensors(batch, device):
    return {k: torch.as_tensor(batch[k], dtype=torch.long if k in {"customers", "devices"} else torch.float32,
                               device=device) for k in ("x", "adj", "static", "customers", "devices")}


def train_encoder(batches, *, seed=53, epochs=50, device="cpu"):
    if type(epochs) is not int or epochs < 1 or epochs > 300:
        raise ValueError("Use 1..300 epochs")
    if device not in {"cpu", "cuda"} or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device unavailable; never substitute fake CUDA results")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    net = FraudNetwork().to(device)
    initial = copy.deepcopy(net.state_dict())
    data = [(tensors(b, device), torch.tensor([e["label"] for e in b["rows"]], device=device, dtype=torch.float32)) for b in batches]
    positives = sum(float(y.sum()) for _, y in data)
    total = sum(len(y) for _, y in data)
    if positives <= 0 or positives >= total:
        raise ValueError("Encoder training needs both outcome classes")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((total-positives)/positives, device=device))
    optimizer = torch.optim.Adam(net.parameters(), lr=.004, weight_decay=.001)
    losses = []
    started = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = sum(criterion(net(b), y)*len(y)/total for b, y in data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if device == "cuda":
        torch.cuda.synchronize()
    changed = sum(float((initial[k].cpu()-v.detach().cpu()).abs().sum()) for k, v in net.state_dict().items() if k.startswith("encoder."))
    return net.eval(), {"device": device, "epochs": epochs, "losses": losses, "training_seconds": time.perf_counter()-started,
                        "encoder_parameter_l1_change": changed, "training_rows": total, "training_positives": int(positives),
                        "trainable_encoder_parameters": sum(p.numel() for p in net.encoder.parameters())}


def embed(net, batches, device="cpu"):
    with torch.inference_mode():
        return [net.transaction_features(tensors(b, device)).cpu().numpy() for b in batches]
