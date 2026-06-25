"""
Train a reduced hybrid QGNN-style FDIA detector.

This is intentionally small and IBM-feasible:
  - select q high-degree buses from the physical grid,
  - use weighted graph diffusion before reduction,
  - encode selected node features into q qubit angles,
  - entangle qubits according to the reduced physical topology,
  - classify from PauliZ expectation values.

The goal is not to beat the full classical CNN on the first run; it is to make
the quantum graph architecture concrete, logged, and comparable with the same
FDIA metrics used by CNN/MLP/GCN/W-GCN/GAT.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pennylane as qml
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RunConfig:
    data: str
    dataset: str
    bus: int
    max_samples: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    test_size: float
    val_size: float
    n_qubits: int
    q_layers: int
    q_device: str
    diff_method: str
    threshold_metric: str
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


def load_ybus(cfg: RunConfig) -> np.ndarray:
    if cfg.dataset == "qgrid":
        return qgrid_ybus(cfg.bus)
    if cfg.dataset == "ruan":
        return ruan_ybus(cfg.bus, cfg.adjacency)
    raise ValueError(f"unknown dataset {cfg.dataset}")


def normalized_adjacency(ybus: np.ndarray, mode: str) -> np.ndarray:
    mag = np.abs(ybus).astype(np.float32)
    if mode == "binary":
        adj = (mag > 1e-12).astype(np.float32)
    elif mode == "weighted":
        adj = mag.copy()
        nz = adj[adj > 1e-12]
        if nz.size:
            adj = adj / float(nz.max())
        adj[adj <= 1e-12] = 0.0
    else:
        raise ValueError(f"unknown adjacency mode {mode}")
    np.fill_diagonal(adj, 0.0)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    deg = adj.sum(axis=1)
    inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    return (inv_sqrt[:, None] * adj * inv_sqrt[None, :]).astype(np.float32)


def select_reduced_nodes(ybus: np.ndarray, n_qubits: int) -> list[int]:
    mag = np.abs(ybus).astype(np.float32)
    np.fill_diagonal(mag, 0.0)
    degree = (mag > 1e-12).sum(axis=1)
    strength = mag.sum(axis=1)
    order = np.lexsort((-strength, -degree))
    return sorted(int(i) for i in order[:n_qubits])


def reduced_edges(ybus: np.ndarray, nodes: list[int]) -> list[tuple[int, int]]:
    node_to_wire = {node: i for i, node in enumerate(nodes)}
    edges = []
    mag = np.abs(ybus)
    for a_idx, a in enumerate(nodes):
        for b in nodes[a_idx + 1 :]:
            if mag[a, b] > 1e-12:
                edges.append((node_to_wire[a], node_to_wire[b]))
    if not edges:
        edges = [(i, (i + 1) % len(nodes)) for i in range(len(nodes))]
    return edges


def reduce_graph_features(x: np.ndarray, adj: np.ndarray, nodes: list[int]) -> np.ndarray:
    diffused = np.einsum("ij,bjf->bif", adj, x)
    return diffused[:, nodes, :].astype(np.float32)


class ReducedQGNNDetector(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        n_qubits: int,
        q_layers: int,
        edges: list[tuple[int, int]],
        q_device: str,
        diff_method: str,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.edges = edges
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_qubits * node_feature_dim, n_qubits),
            nn.Tanh(),
        )
        dev = qml.device(q_device, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for wire in range(n_qubits):
                qml.RY(np.pi * inputs[wire], wires=wire)
            for a, b in edges:
                qml.CNOT(wires=[a, b])
            for layer in range(q_layers):
                for wire in range(n_qubits):
                    qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
                for a, b in edges:
                    qml.CNOT(wires=[a, b])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (q_layers, n_qubits, 3)}
        self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.encoder(x)
        if angles.ndim == 1:
            q = self.quantum(angles)
        else:
            q = torch.stack([self.quantum(row) for row in angles], dim=0)
        return self.head(q).squeeze(1)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def module_parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def write_architecture_summary(
    model: ReducedQGNNDetector,
    cfg: RunConfig,
    nodes: list[int],
    edges: list[tuple[int, int]],
    out_dir: Path,
):
    quantum_weight_shape = [cfg.q_layers, cfg.n_qubits, 3]
    summary = {
        "model": "qgnn",
        "dataset": cfg.dataset,
        "bus": cfg.bus,
        "n_qubits": cfg.n_qubits,
        "q_layers": cfg.q_layers,
        "q_device": cfg.q_device,
        "diff_method": cfg.diff_method,
        "selected_nodes": nodes,
        "reduced_edges": edges,
        "node_feature_dim": model.node_feature_dim,
        "adjacency_mode": cfg.adjacency_mode,
        "encoding": "graph-diffused selected node features -> tanh linear encoder -> RY angles",
        "circuit_sequence": [
            "RY(pi * encoded_feature_i) on each selected-node qubit",
            "CNOT entanglers over reduced physical topology",
            "for each quantum layer: Rot(phi, theta, omega) on every qubit",
            "for each quantum layer: repeat CNOT entanglers over reduced physical topology",
            "measure PauliZ expectation on every qubit",
        ],
        "entangler": "CNOT gates on reduced physical topology, ring fallback if disconnected",
        "readout": "PauliZ expectations on all qubits -> classical binary head",
        "quantum_weight_shape": quantum_weight_shape,
        "quantum_parameters": int(np.prod(quantum_weight_shape)),
        "classical_encoder_parameters": module_parameter_count(model.encoder),
        "classical_head_parameters": module_parameter_count(model.head),
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
            "n_qubits",
            "q_layers",
            "q_device",
            "diff_method",
            "selected_nodes",
            "reduced_edges",
            "node_feature_dim",
            "adjacency_mode",
            "quantum_weight_shape",
            "quantum_parameters",
            "classical_encoder_parameters",
            "classical_head_parameters",
            "total_parameters",
            "trainable_parameters",
            "encoding",
            "entangler",
            "readout",
        ]:
            f.write(f"{key}: {summary[key]}\n")
        f.write("circuit_sequence:\n")
        for step in summary["circuit_sequence"]:
            f.write(f"- {step}\n")
        f.write("\n")
        f.write(summary["architecture"])
        f.write("\n")
    return summary


def collect_probabilities(model, loader, device):
    model.eval()
    probs, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(labels), np.concatenate(probs), elapsed


def metrics_from_probabilities(y: np.ndarray, p: np.ndarray, threshold: float, elapsed: float):
    pred = (p >= threshold).astype(np.int64)
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
        "threshold": float(threshold),
    }


def select_threshold(y: np.ndarray, p: np.ndarray, metric: str):
    candidates = np.unique(np.concatenate(([0.5], p.astype(np.float64))))
    best_threshold = 0.5
    best_metrics = metrics_from_probabilities(y, p, best_threshold, elapsed=0.0)
    best_score = float(best_metrics[metric])
    best_tie = float(best_metrics["mcc"])
    for threshold in candidates:
        metrics = metrics_from_probabilities(y, p, float(threshold), elapsed=0.0)
        score = float(metrics[metric])
        tie = float(metrics["mcc"])
        if score > best_score or (score == best_score and tie > best_tie):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
            best_tie = tie
    return best_threshold, best_metrics


def evaluate(model, loader, device, threshold=None, threshold_metric: str = "f1"):
    y, p, elapsed = collect_probabilities(model, loader, device)
    if threshold is None:
        threshold, metrics = select_threshold(y, p, threshold_metric)
        metrics["latency_ms_per_sample"] = 1000.0 * elapsed / max(len(y), 1)
        return metrics
    return metrics_from_probabilities(y, p, threshold, elapsed)


def train(cfg: RunConfig, out_dir: Path):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"loading {cfg.data}", flush=True)
    x, y = load_detector_data(Path(cfg.data), cfg.bus, cfg.max_samples, cfg.seed)
    ybus = load_ybus(cfg)
    adj = normalized_adjacency(ybus, cfg.adjacency_mode)
    nodes = select_reduced_nodes(ybus, cfg.n_qubits)
    edges = reduced_edges(ybus, nodes)
    x = reduce_graph_features(x, adj, nodes)
    print(
        f"loaded reduced x={x.shape} y={y.shape} positives={int(y.sum())} "
        f"selected_nodes={nodes} reduced_edges={edges}",
        flush=True,
    )

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

    # PennyLane TorchLayer is CPU-friendly and avoids GPU/autograd plugin surprises.
    device = torch.device("cpu")
    model = ReducedQGNNDetector(
        node_feature_dim=x_train.shape[-1],
        n_qubits=cfg.n_qubits,
        q_layers=cfg.q_layers,
        edges=edges,
        q_device=cfg.q_device,
        diff_method=cfg.diff_method,
    ).to(device)
    arch = write_architecture_summary(model, cfg, nodes, edges, out_dir)
    print(
        f"training qgnn on {device} | total params={arch['total_parameters']} "
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
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, device, threshold=None, threshold_metric=cfg.threshold_metric)
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
    val_final = evaluate(model, val_loader, device, threshold=None, threshold_metric=cfg.threshold_metric)
    calibrated_threshold = float(val_final["threshold"])
    test = evaluate(model, test_loader, device, threshold=calibrated_threshold, threshold_metric=cfg.threshold_metric)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                **asdict(cfg),
                "input_dim": int(x.shape[1] * x.shape[2]),
                "selected_nodes": nodes,
                "reduced_edges": edges,
                "node_feature_dim": int(x.shape[2]),
                "device": str(device),
                "total_parameters": arch["total_parameters"],
                "trainable_parameters": arch["trainable_parameters"],
                "threshold_calibration": cfg.threshold_metric,
                "calibrated_threshold": calibrated_threshold,
            },
            f,
            indent=2,
        )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(test, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "selected_nodes": nodes,
            "reduced_edges": edges,
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
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--q-layers", type=int, default=2)
    ap.add_argument("--q-device", default="default.qubit")
    ap.add_argument("--diff-method", default="backprop")
    ap.add_argument("--threshold-metric", choices=["f1", "balanced_accuracy", "mcc"], default="f1")
    ap.add_argument("--adjacency", default="external/ruan_fdia")
    ap.add_argument("--adjacency-mode", choices=["binary", "weighted"], default="weighted")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg_args = vars(args).copy()
    cfg_args.pop("out")
    cfg = RunConfig(**cfg_args)
    out = Path(args.out or f"runs/detectors/{args.dataset}{args.bus}_qgnn_{args.max_samples}")
    train(cfg, out)


if __name__ == "__main__":
    main()
