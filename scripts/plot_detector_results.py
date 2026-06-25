"""
Aggregate and plot detector results.

Reads detector run directories containing:
  - metrics.json
  - history.csv

Writes:
  - detector_metrics_summary.csv
  - detector_metrics_bars.png
  - detector_learning_curves.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(r"(?P<dataset>qgrid|ruan)(?P<bus>\d+)_?(?P<model>cnn1d|mlp).*")


def parse_run_name(path: Path):
    m = RUN_RE.match(path.name)
    if not m:
        return {"dataset": "unknown", "bus": "unknown", "model": path.name}
    d = m.groupdict()
    return {"dataset": d["dataset"], "bus": int(d["bus"]), "model": d["model"]}


def load_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({"run": run_dir.name, **parse_run_name(run_dir), **metrics})
    if not rows:
        raise FileNotFoundError(f"no metrics.json files found under {root}")
    return pd.DataFrame(rows).sort_values(["dataset", "bus", "model"])


def load_histories(root: Path) -> pd.DataFrame:
    rows = []
    for hist_path in sorted(root.glob("*/history.csv")):
        run_dir = hist_path.parent
        df = pd.read_csv(hist_path)
        meta = parse_run_name(run_dir)
        df.insert(0, "run", run_dir.name)
        for k, v in meta.items():
            df.insert(1, k, v)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no history.csv files found under {root}")
    return pd.concat(rows, ignore_index=True)


def plot_bars(metrics: pd.DataFrame, out: Path):
    metrics = metrics.copy()
    metrics["label"] = metrics["dataset"].astype(str) + "-" + metrics["bus"].astype(str)
    metrics = metrics.sort_values(["dataset", "bus"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    good = ["f1", "auroc", "auprc", "balanced_accuracy", "mcc"]
    bad = ["fpr", "fnr", "latency_ms_per_sample"]

    metrics.plot(x="label", y=good, kind="bar", ax=axes[0], width=0.82)
    axes[0].set_title("Detector Quality Metrics")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("score")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    metrics.plot(x="label", y=bad, kind="bar", ax=axes[1], width=0.82)
    axes[1].set_title("Operational Error / Runtime Metrics")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("rate or ms/sample")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_learning(hist: pd.DataFrame, out: Path):
    hist = hist.copy()
    hist["label"] = hist["dataset"].astype(str) + "-" + hist["bus"].astype(str)
    panels = [("val_f1", "Validation F1"), ("val_auprc", "Validation AUPRC"), ("train_loss", "Train Loss")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, panels):
        for label, grp in hist.groupby("label"):
            grp = grp.sort_values("epoch")
            ax.plot(grp["epoch"], grp[col], marker="o", linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/detectors")
    ap.add_argument("--out-dir", default="runs/plots")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(root)
    hist = load_histories(root)
    metrics.to_csv(out_dir / "detector_metrics_summary.csv", index=False)
    plot_bars(metrics, out_dir / "detector_metrics_bars.png")
    plot_learning(hist, out_dir / "detector_learning_curves.png")
    print(metrics[["dataset", "bus", "model", "f1", "auroc", "auprc", "fpr", "fnr", "latency_ms_per_sample"]])
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
