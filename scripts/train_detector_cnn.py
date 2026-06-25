"""
Train published-style deep FDIA detector baselines on QGrid-Synth.

Models:
  - mlp: dense measurement-vector detector
  - cnn1d: 1D convolutional detector over ordered measurement features

The script reads QGrid-Synth Parquet files produced by generate_dataset.py and
reports detector metrics used in FDIA literature: F1, AUROC, AUPRC, MCC, FPR,
FNR, balanced accuracy, and latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
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
    model: str
    max_samples: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    test_size: float
    val_size: float
    feature_mode: str


def _stack_list_column(table, name: str) -> np.ndarray:
    col = table[name].combine_chunks()
    return np.asarray(col.to_pylist(), dtype=np.float32)


def load_qgrid(path: Path, max_samples: int, feature_mode: str, seed: int):
    table = pq.read_table(path, columns=["label", "z", "a", "sample_id", "attack_type"])
    y = np.asarray(table["label"].combine_chunks().to_numpy(), dtype=np.int64)
    z = _stack_list_column(table, "z")

    if feature_mode == "z":
        x = z
    elif feature_mode == "z_plus_abs_a":
        a = np.abs(_stack_list_column(table, "a"))
        x = np.concatenate([z, a], axis=1)
    else:
        raise ValueError(f"unknown feature_mode {feature_mode}")

    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        x, y = x[idx], y[idx]
    return x, y


class MLPDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN1DDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1))).squeeze(1)


def make_model(name: str, input_dim: int) -> nn.Module:
    if name == "mlp":
        return MLPDetector(input_dim)
    if name == "cnn1d":
        return CNN1DDetector(input_dim)
    raise ValueError(f"unknown model {name}")


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.numpy())
    elapsed = time.perf_counter() - t0
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    pred = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
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
    return out


def train(cfg: RunConfig, out_dir: Path):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    print(f"loading {cfg.data}", flush=True)
    x, y = load_qgrid(Path(cfg.data), cfg.max_samples, cfg.feature_mode, cfg.seed)
    print(f"loaded x={x.shape} y={y.shape} positives={int(y.sum())}", flush=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=rel_val, random_state=cfg.seed, stratify=y_train
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"training {cfg.model} on {device}", flush=True)
    model = make_model(cfg.model, x_train.shape[1]).to(device)
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
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
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
    test = evaluate(model, test_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**asdict(cfg), "input_dim": int(x.shape[1]), "device": str(device)}, f, indent=2)
    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(test, f, indent=2)
    torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, out_dir / "model.pt")
    print("\nTEST")
    for k, v in test.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["mlp", "cnn1d"], default="cnn1d")
    ap.add_argument("--max-samples", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--feature-mode", choices=["z", "z_plus_abs_a"], default="z")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg_args = vars(args).copy()
    cfg_args.pop("out")
    cfg = RunConfig(**cfg_args)
    out = Path(args.out or f"runs/detectors/{Path(args.data).stem}_{args.model}_{args.max_samples}")
    train(cfg, out)


if __name__ == "__main__":
    main()
