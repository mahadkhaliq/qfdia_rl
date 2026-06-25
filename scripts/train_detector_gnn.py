"""
Train graph neural FDIA detector baselines on QGrid-Synth and Ruan CAISO.

This is a lightweight pure-PyTorch GCN/GAT baseline. It uses the grid topology from:
  - Pandapower Ybus for QGrid-Synth IEEE 30/57/118 systems
  - Ruan/STGDL admittance matrices for Ruan CAISO 30/118 systems

Each sample is reshaped from a measurement vector z into node features:
  QGrid-Synth: [P, Q, Vm, theta] per bus
  Ruan CAISO:  [Vm, theta] per bus
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunConfig:
    data: str
    dataset: str
    bus: int
    model: str
    max_samples: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    test_size: float
    val_size: float
    feature_mode: str
    adjacency: str
    adjacency_mode: str


def _stack_list_column(table, name: str) -> np.ndarray:
    col = table[name].combine_chunks()
    return np.asarray(col.to_pylist(), dtype=np.float32)


def load_detector_data(path: Path, bus: int, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["label", "z"])
    y = np.asarray(table["label"].combine_chunks().to_numpy(), dtype=np.int64)
    z = _stack_list_column(table, "z")
    if z.shape[1] % bus != 0:
        raise ValueError(f"z dimension {z.shape[1]} is not divisible by bus count {bus}")
    x = z.reshape(len(z), z.shape[1] // bus, bus).transpose(0, 2, 1)
    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        x, y = x[idx], y[idx]
    return x.astype(np.float32), y


def qgrid_ybus(bus: int) -> np.ndarray:
    import pandapower as pp
    import pandapower.networks as pn

    loader = {30: pn.case30, 57: pn.case57, 118: pn.case118}.get(bus)
    if loader is None:
        raise ValueError(f"unsupported QGrid bus count {bus}")
    net = loader()
    pp.runpp(net, calculate_voltage_angles=True, init="dc", numba=False)
    return np.asarray(net._ppc["internal"]["Ybus"].todense())


def ruan_ybus(bus: int, adjacency_path: str) -> np.ndarray:
    path = Path(adjacency_path)
    if path.is_dir():
        path = path / f"{bus}-bus" / "extracted" / f"AdmittanceMatrix_{bus}.mat"
    mat = loadmat(path)
    if "G" not in mat or "B" not in mat:
        raise KeyError(f"{path} must contain G and B admittance matrices")
    return np.asarray(mat["G"], dtype=float) + 1j * np.asarray(mat["B"], dtype=float)


def normalized_adjacency(ybus: np.ndarray, mode: str = "binary") -> np.ndarray:
    mag = np.abs(ybus).astype(np.float32)
    if mode == "binary":
        adj = (mag > 1e-12).astype(np.float32)
    elif mode == "weighted":
        adj = mag.copy()
        nonzero = adj[adj > 1e-12]
        if nonzero.size:
            adj = adj / float(nonzero.max())
        adj[adj <= 1e-12] = 0.0
    else:
        raise ValueError(f"unknown adjacency mode {mode}")
    np.fill_diagonal(adj, 0.0)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    deg = adj.sum(axis=1)
    inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    return (inv_sqrt[:, None] * adj * inv_sqrt[None, :]).astype(np.float32)


def load_adjacency(cfg: RunConfig) -> tuple[np.ndarray, int]:
    if cfg.dataset == "qgrid":
        ybus = qgrid_ybus(cfg.bus)
    elif cfg.dataset == "ruan":
        ybus = ruan_ybus(cfg.bus, cfg.adjacency)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset}")
    edge_count = int(np.count_nonzero(np.triu((np.abs(ybus) > 1e-12), k=1)))
    return normalized_adjacency(ybus, cfg.adjacency_mode), edge_count


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,bjf->bif", adj, x)
        return self.linear(x)


class GCNDetector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm1(self.conv1(x, adj)))
        x = self.dropout(x)
        x = torch.relu(self.norm2(self.conv2(x, adj)))
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        graph = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(graph).squeeze(1)


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        bias_dim = heads * out_dim if concat else out_dim
        self.bias = nn.Parameter(torch.zeros(bias_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        h = self.linear(x).view(batch, nodes, self.heads, self.out_dim)
        h = h.permute(0, 2, 1, 3)  # (batch, heads, nodes, out_dim)
        src = (h * self.attn_src.view(1, self.heads, 1, self.out_dim)).sum(dim=-1)
        dst = (h * self.attn_dst.view(1, self.heads, 1, self.out_dim)).sum(dim=-1)
        scores = self.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2))
        mask = adj > 0
        scores = scores.masked_fill(~mask.view(1, 1, nodes, nodes), torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, h)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(batch, nodes, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class GATDetector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        head_dim = hidden_dim // heads
        self.attn1 = GraphAttentionLayer(in_dim, head_dim, heads=heads, concat=True, dropout=dropout)
        self.attn2 = GraphAttentionLayer(hidden_dim, head_dim, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm1(self.attn1(x, adj)))
        x = self.dropout(x)
        x = torch.relu(self.norm2(self.attn2(x, adj)))
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        graph = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(graph).squeeze(1)


def make_model(name: str, feature_dim: int) -> nn.Module:
    if name in {"gcn", "wgcn"}:
        return GCNDetector(feature_dim)
    if name == "gat":
        return GATDetector(feature_dim)
    raise ValueError(f"unknown model {name}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def write_architecture_summary(model: nn.Module, cfg: RunConfig, feature_dim: int, nodes: int, edges: int, out_dir: Path):
    summary = {
        "model": cfg.model,
        "dataset": cfg.dataset,
        "bus": cfg.bus,
        "nodes": int(nodes),
        "edges": int(edges),
        "node_feature_dim": int(feature_dim),
        "feature_mode": cfg.feature_mode,
        "adjacency_mode": cfg.adjacency_mode,
        **parameter_counts(model),
        "architecture": repr(model),
    }
    with open(out_dir / "architecture.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "architecture.txt", "w") as f:
        for key in [
            "model",
            "dataset",
            "bus",
            "nodes",
            "edges",
            "node_feature_dim",
            "feature_mode",
            "adjacency_mode",
            "total_parameters",
            "trainable_parameters",
        ]:
            f.write(f"{key}: {summary[key]}\n")
        f.write("\n")
        f.write(summary["architecture"])
        f.write("\n")
    return summary


def evaluate(model, loader, adj, device):
    model.eval()
    probs, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb, adj)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.numpy())
    elapsed = time.perf_counter() - t0
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    pred = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "latency_ms_per_sample": 1000.0 * elapsed / max(len(y), 1),
        "n": int(len(y)),
    }


def train(cfg: RunConfig, out_dir: Path):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"loading {cfg.data}", flush=True)
    x, y = load_detector_data(Path(cfg.data), cfg.bus, cfg.max_samples, cfg.seed)
    adj_np, edge_count = load_adjacency(cfg)
    if adj_np.shape[0] != x.shape[1]:
        raise ValueError(f"adjacency nodes {adj_np.shape[0]} != data nodes {x.shape[1]}")
    print(f"loaded x={x.shape} y={y.shape} positives={int(y.sum())} edges={edge_count}", flush=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=rel_val, random_state=cfg.seed, stratify=y_train
    )

    scaler = StandardScaler()
    train_flat = x_train.reshape(-1, x_train.shape[-1])
    scaler.fit(train_flat)
    x_train = scaler.transform(train_flat).reshape(x_train.shape).astype(np.float32)
    x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape).astype(np.float32)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape).astype(np.float32)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    adj = torch.from_numpy(adj_np).to(device)
    print(f"training {cfg.model} on {device}", flush=True)
    model = make_model(cfg.model, x_train.shape[-1]).to(device)
    arch = write_architecture_summary(model, cfg, x_train.shape[-1], x_train.shape[1], edge_count, out_dir)
    print(
        f"architecture parameters: total={arch['total_parameters']} "
        f"trainable={arch['trainable_parameters']}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.float32))),
        batch_size=cfg.batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.float32))),
        batch_size=cfg.batch_size,
    )

    best_state = None
    best_val = -1.0
    history = []
    history_path = out_dir / "history.csv"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, adj), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, adj, device)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        if val["f1"] > best_val:
            best_val = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:03d}/{cfg.epochs} loss {rec['train_loss']:.4f} "
            f"val_f1 {val['f1']:.4f} val_auprc {val['auprc']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, adj, device)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                **asdict(cfg),
                "input_dim": int(x.shape[1] * x.shape[2]),
                "nodes": int(x.shape[1]),
                "node_feature_dim": int(x.shape[2]),
                "edges": int(edge_count),
                "device": str(device),
                "total_parameters": arch["total_parameters"],
                "trainable_parameters": arch["trainable_parameters"],
            },
            f,
            indent=2,
        )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(test, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "adjacency": adj_np,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        out_dir / "model.pt",
    )
    print("\nTEST")
    for k, v in test.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", choices=["qgrid", "ruan"], required=True)
    ap.add_argument("--bus", type=int, required=True)
    ap.add_argument("--model", choices=["gcn", "wgcn", "gat"], default="gcn")
    ap.add_argument("--max-samples", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--feature-mode", choices=["z"], default="z")
    ap.add_argument("--adjacency", default="external/ruan_fdia")
    ap.add_argument("--adjacency-mode", choices=["binary", "weighted"], default="binary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg_args = vars(args).copy()
    cfg_args.pop("out")
    cfg = RunConfig(**cfg_args)
    out = Path(args.out or f"runs/detectors/{args.dataset}{args.bus}_{args.model}_{args.max_samples}")
    train(cfg, out)


if __name__ == "__main__":
    main()
