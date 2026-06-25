"""
Sweep decision thresholds for a trained reduced QGNN detector.

This is useful for paper analysis because a single threshold can hide the
precision/recall/FPR tradeoff. The script reloads a QGNN run directory,
reconstructs the same test split, and writes:

  - threshold_sweep.csv
  - threshold_sweep_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_detector_qgnn import (  # noqa: E402
    ReducedQGNNDetector,
    RunConfig,
    load_detector_data,
    load_ybus,
    metrics_from_probabilities,
    normalized_adjacency,
    reduce_graph_features,
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def reconstruct_test_data(cfg: RunConfig, run_config: dict, checkpoint: dict):
    x, y = load_detector_data(Path(cfg.data), cfg.bus, cfg.max_samples, cfg.seed)
    ybus = load_ybus(cfg)
    adj = normalized_adjacency(ybus, cfg.adjacency_mode)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    train_test_split(x_train, y_train, test_size=rel_val, random_state=cfg.seed, stratify=y_train)

    nodes = run_config["selected_nodes"]
    x_test = reduce_graph_features(x_test, adj, nodes, cfg.feature_mode)
    flat = x_test.reshape(-1, x_test.shape[-1])
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    x_test = ((flat - mean) / scale).reshape(x_test.shape).astype(np.float32)
    return x_test, y_test


def collect_probs(model: ReducedQGNNDetector, x: np.ndarray, batch_size: int):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size])
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def sweep_thresholds(y: np.ndarray, p: np.ndarray, n_thresholds: int):
    thresholds = np.unique(np.concatenate(([0.5], np.linspace(0.0, 1.0, n_thresholds), p.astype(np.float64))))
    rows = []
    for threshold in thresholds:
        metrics = metrics_from_probabilities(y, p, float(threshold), elapsed=0.0)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def best_by(rows: list[dict], key: str):
    return max(rows, key=lambda r: (float(r[key]), float(r["mcc"]), float(r["balanced_accuracy"])))


def best_recall_at_fpr(rows: list[dict], max_fpr: float):
    feasible = [r for r in rows if float(r["fpr"]) <= max_fpr]
    if not feasible:
        return None
    return max(feasible, key=lambda r: (float(r["recall"]), float(r["f1"]), float(r["mcc"])))


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-thresholds", type=int, default=201)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_config = load_json(run_dir / "config.json")
    cfg_fields = RunConfig.__dataclass_fields__.keys()
    cfg = RunConfig(**{key: run_config[key] for key in cfg_fields})
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")

    x_test, y_test = reconstruct_test_data(cfg, run_config, checkpoint)
    model = ReducedQGNNDetector(
        node_feature_dim=int(run_config["node_feature_dim"]),
        n_qubits=int(run_config["n_qubits"]),
        q_layers=int(run_config["q_layers"]),
        edges=[tuple(edge) for edge in run_config["reduced_edges"]],
        q_device=run_config.get("q_device", "default.qubit"),
        diff_method=run_config.get("diff_method", "backprop"),
    )
    model.load_state_dict(checkpoint["model"])
    probs = collect_probs(model, x_test, args.batch_size)

    rows = sweep_thresholds(y_test, probs, args.n_thresholds)
    write_csv(run_dir / "threshold_sweep.csv", rows)
    summary = {
        "run_dir": str(run_dir),
        "n": int(len(y_test)),
        "best_f1": best_by(rows, "f1"),
        "best_balanced_accuracy": best_by(rows, "balanced_accuracy"),
        "best_mcc": best_by(rows, "mcc"),
        "best_recall_at_fpr_0.05": best_recall_at_fpr(rows, 0.05),
        "best_recall_at_fpr_0.10": best_recall_at_fpr(rows, 0.10),
        "best_recall_at_fpr_0.20": best_recall_at_fpr(rows, 0.20),
    }
    with open(run_dir / "threshold_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
