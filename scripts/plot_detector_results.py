"""
Aggregate and plot detector results.

Reads detector run directories containing:
  - metrics.json
  - history.csv

Writes:
  - detector_metrics_summary.csv
  - detector_metrics_bars.png
  - detector_learning_curves.png
  - metrics/*.png
  - learning_curves/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(r"(?P<dataset>qgrid|ruan)(?P<bus>\d+)_?(?P<model>cnn1d|mlp|gcn|wgcn|gat|qgnn).*")
DATASET_NAMES = {
    "qgrid": "QGrid-Synth",
    "ruan": "Ruan CAISO",
}
MODEL_NAMES = {
    "cnn1d": "1D-CNN",
    "mlp": "MLP",
    "gcn": "GCN",
    "wgcn": "W-GCN",
    "gat": "GAT",
    "qgnn": "QGNN",
}
METRIC_TITLES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "mcc": "Matthews Correlation Coefficient",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "fpr": "False Positive Rate",
    "fnr": "False Negative Rate",
    "latency_ms_per_sample": "Latency",
}
METRIC_YLABELS = {
    "latency_ms_per_sample": "ms/sample",
}
SORT_COLS = ["dataset_order", "bus", "model_order", "variant_order", "run"]
PAPER_SCORE_METRICS = ["f1", "auroc", "auprc"]
PAPER_MODEL_ORDER = [
    "1D-CNN",
    "MLP",
    "GCN",
    "W-GCN",
    "GAT",
    "QGNN-pilot",
    "QGNN-cal",
    "QGNN-enh4",
    "QGNN-enh6",
    "QGNN-enh6-BA",
]
MODEL_COLORS = {
    "1D-CNN": "#2f6f9f",
    "MLP": "#4c956c",
    "GCN": "#d68c45",
    "W-GCN": "#8f5f9f",
    "GAT": "#b5525c",
    "QGNN-pilot": "#595959",
    "QGNN-cal": "#756bb1",
    "QGNN-enh4": "#9e77ad",
    "QGNN-enh6": "#6b4c9a",
    "QGNN-enh6-BA": "#4a3474",
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_run_name(path: Path):
    m = RUN_RE.match(path.name)
    if not m:
        return {"dataset": "unknown", "bus": "unknown", "model": path.name, "variant": "run"}
    d = m.groupdict()
    variant = "full"
    if "enhanced6_balacc" in path.name:
        variant = "enhanced6_balacc"
    elif "enhanced6" in path.name:
        variant = "enhanced6"
    elif "enhanced4" in path.name:
        variant = "enhanced4"
    elif "zplusa" in path.name or "z_plus_abs_a" in path.name:
        variant = "z_plus_abs_a"
    elif "calibrated" in path.name:
        variant = "calibrated"
    elif "pilot" in path.name:
        variant = "pilot"
    return {"dataset": d["dataset"], "bus": int(d["bus"]), "model": d["model"], "variant": variant}


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_name"] = df["dataset"].map(DATASET_NAMES).fillna(df["dataset"].astype(str))
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"].astype(str))
    qgnn = df["model"].eq("qgnn")
    df.loc[qgnn & df["variant"].eq("pilot"), "model_name"] = "QGNN-pilot"
    df.loc[qgnn & df["variant"].eq("calibrated"), "model_name"] = "QGNN-cal"
    df.loc[qgnn & df["variant"].eq("enhanced4"), "model_name"] = "QGNN-enh4"
    df.loc[qgnn & df["variant"].eq("enhanced6"), "model_name"] = "QGNN-enh6"
    df.loc[qgnn & df["variant"].eq("enhanced6_balacc"), "model_name"] = "QGNN-enh6-BA"
    residual = df["variant"].eq("z_plus_abs_a") & ~qgnn
    df.loc[residual, "model_name"] = df.loc[residual, "model_name"] + "+|a|"
    df["model_family"] = df["model"].map(MODEL_NAMES).fillna(df["model"].astype(str))
    df["is_oracle"] = df["variant"].eq("z_plus_abs_a")
    df["system_label"] = df["dataset_name"] + " " + df["bus"].astype(str) + "-bus"
    df["system_key"] = df["dataset"].astype(str) + "_" + df["bus"].astype(str)
    df["plot_label"] = df["system_label"] + "\n" + df["model_name"]
    df["dataset_order"] = df["dataset"].map({"qgrid": 0, "ruan": 1}).fillna(99)
    df["model_order"] = df["model"].map(
        {"cnn1d": 0, "mlp": 1, "gcn": 2, "wgcn": 3, "gat": 4, "qgnn": 5}
    ).fillna(99)
    df["variant_order"] = df["variant"].map(
        {
            "full": 0,
            "z_plus_abs_a": 1,
            "pilot": 2,
            "calibrated": 3,
            "enhanced4": 4,
            "enhanced6": 5,
            "enhanced6_balacc": 6,
        }
    ).fillna(9)
    return df


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d0d0d0",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def load_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({"run": run_dir.name, **parse_run_name(run_dir), **metrics})
    if not rows:
        raise FileNotFoundError(f"no metrics.json files found under {root}")
    df = add_display_columns(pd.DataFrame(rows))
    return df.sort_values(SORT_COLS)


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
    return add_display_columns(pd.concat(rows, ignore_index=True))


def plot_bars(metrics: pd.DataFrame, out: Path):
    metrics = metrics.copy()
    metrics = metrics.sort_values(SORT_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    good = ["f1", "auroc", "auprc", "balanced_accuracy", "mcc"]
    bad = ["fpr", "fnr", "latency_ms_per_sample"]

    metrics.plot(x="plot_label", y=good, kind="bar", ax=axes[0], width=0.82)
    axes[0].set_title("FDIA Detection Quality by Model")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("score")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    metrics.plot(x="plot_label", y=bad, kind="bar", ax=axes[1], width=0.82)
    axes[1].set_title("FDIA Detection Error / Runtime by Model")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("rate or ms/sample")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_individual_metric_bars(metrics: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics.sort_values(SORT_COLS)
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "auroc",
        "auprc",
        "fpr",
        "fnr",
        "latency_ms_per_sample",
    ]
    for col in metric_cols:
        if col not in metrics.columns:
            continue
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        metrics.plot(x="plot_label", y=col, kind="bar", ax=ax, legend=False, width=0.78)
        ax.set_title(f"{METRIC_TITLES.get(col, col)} by Dataset and Detector")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_YLABELS.get(col, "score"))
        if col != "latency_ms_per_sample":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=180)
        plt.close(fig)


def plot_learning(hist: pd.DataFrame, out: Path):
    hist = hist.copy()
    hist = hist.sort_values(SORT_COLS + ["epoch"])
    panels = [("val_f1", "Validation F1"), ("val_auprc", "Validation AUPRC"), ("train_loss", "Train Loss")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
    for ax, (col, title) in zip(axes, panels):
        for label, grp in hist.groupby("plot_label", sort=False):
            grp = grp.sort_values("epoch")
            ax.plot(grp["epoch"], grp[col], marker="o", linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_individual_learning_curves(hist: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = hist.sort_values(SORT_COLS + ["epoch"])
    curves = [
        ("val_f1", "Validation F1", "score"),
        ("val_auprc", "Validation AUPRC", "score"),
        ("val_auroc", "Validation AUROC", "score"),
        ("train_loss", "Training Loss", "loss"),
    ]
    for col, title, ylabel in curves:
        if col not in hist.columns:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
        for label, grp in hist.groupby("plot_label", sort=False):
            grp = grp.sort_values("epoch")
            ax.plot(grp["epoch"], grp[col], marker="o", linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if ylabel == "score":
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
        fig.savefig(out_dir / f"{col}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_group_metric_summary(metrics: pd.DataFrame, out: Path, title: str, x_col: str):
    metrics = metrics.sort_values(SORT_COLS)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    good = ["f1", "auroc", "auprc", "balanced_accuracy", "mcc"]
    bad = ["fpr", "fnr", "latency_ms_per_sample"]

    metrics.plot(x=x_col, y=good, kind="bar", ax=axes[0], width=0.78)
    axes[0].set_title(f"{title}: Detection Quality")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("score")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    metrics.plot(x=x_col, y=bad, kind="bar", ax=axes[1], width=0.78)
    axes[1].set_title(f"{title}: Error / Runtime")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("rate or ms/sample")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_group_learning(hist: pd.DataFrame, out: Path, title: str, label_col: str):
    hist = hist.sort_values(SORT_COLS + ["epoch"])
    panels = [("val_f1", "Validation F1"), ("val_auprc", "Validation AUPRC"), ("train_loss", "Train Loss")]
    n_series = max(1, hist[label_col].nunique())
    fig_height = 5.3 if n_series <= 4 else 6.3
    fig, axes = plt.subplots(1, 3, figsize=(15.8, fig_height), constrained_layout=True)
    for ax, (col, panel_title) in zip(axes, panels):
        if col not in hist.columns:
            ax.axis("off")
            continue
        for label, grp in hist.groupby(label_col, sort=False):
            grp = grp.sort_values("epoch")
            ax.plot(grp["epoch"], grp[col], marker="o", linewidth=2, label=label)
        ax.set_title(panel_title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="center left", ncol=1, bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title, y=1.02)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_views(metrics: pd.DataFrame, hist: pd.DataFrame, out_dir: Path):
    by_system = out_dir / "by_system"
    by_bus = out_dir / "by_bus"
    by_bus_model = out_dir / "by_bus_model"
    by_system.mkdir(parents=True, exist_ok=True)
    by_bus.mkdir(parents=True, exist_ok=True)
    by_bus_model.mkdir(parents=True, exist_ok=True)

    for system_key, grp in metrics.groupby("system_key", sort=False):
        label = str(grp["system_label"].iloc[0])
        stem = slugify(system_key)
        plot_group_metric_summary(grp, by_system / f"{stem}_metrics.png", label, "model_name")
        hist_grp = hist[hist["system_key"].eq(system_key)]
        if not hist_grp.empty:
            plot_group_learning(hist_grp, by_system / f"{stem}_learning.png", label, "model_name")

    for bus, grp in metrics.groupby("bus", sort=True):
        title = f"IEEE {bus}-bus"
        stem = f"{bus}_bus"
        plot_group_metric_summary(grp, by_bus / f"{stem}_metrics.png", title, "plot_label")
        hist_grp = hist[hist["bus"].eq(bus)]
        if not hist_grp.empty:
            plot_group_learning(hist_grp, by_bus / f"{stem}_learning.png", title, "plot_label")

    for (bus, model), grp in metrics.groupby(["bus", "model"], sort=True):
        family = str(grp["model_family"].iloc[0])
        title = f"IEEE {bus}-bus: {family}"
        stem = f"{bus}_bus_{slugify(str(model))}"
        plot_group_metric_summary(grp, by_bus_model / f"{stem}_metrics.png", title, "plot_label")
        hist_grp = hist[hist["bus"].eq(bus) & hist["model"].eq(model)]
        if not hist_grp.empty:
            plot_group_learning(hist_grp, by_bus_model / f"{stem}_learning.png", title, "plot_label")


def ordered_system_labels(metrics: pd.DataFrame) -> list[str]:
    cols = ["system_label", "dataset_order", "bus"]
    return (
        metrics[cols]
        .drop_duplicates()
        .sort_values(["dataset_order", "bus", "system_label"])["system_label"]
        .tolist()
    )


def ordered_model_labels(metrics: pd.DataFrame) -> list[str]:
    present = set(metrics["model_name"].astype(str))
    ordered = [name for name in PAPER_MODEL_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def deployable_only(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[~metrics["is_oracle"]].copy()


def annotate_heatmap(ax, values: pd.DataFrame):
    for row_idx, (_, row) in enumerate(values.iterrows()):
        for col_idx, value in enumerate(row):
            if pd.isna(value):
                ax.text(col_idx, row_idx, "n/a", ha="center", va="center", color="#8a8a8a", fontsize=7)
                continue
            color = "white" if value < 0.55 else "#1c1c1c"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)


def plot_metric_heatmap(metrics: pd.DataFrame, metric: str, out: Path, title: str):
    systems = ordered_system_labels(metrics)
    models = ordered_model_labels(metrics)
    pivot = (
        metrics.pivot_table(index="system_label", columns="model_name", values=metric, aggfunc="max")
        .reindex(index=systems, columns=models)
    )
    width = max(9.0, 1.15 * len(models))
    height = max(4.2, 0.62 * len(systems) + 1.6)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f2f2f2")
    image = ax.imshow(pivot.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    annotate_heatmap(ax, pivot)
    ax.set_xticks(range(len(models)), labels=models, rotation=35, ha="right")
    ax.set_yticks(range(len(systems)), labels=systems)
    ax.set_title(title)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Dataset / system")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(METRIC_TITLES.get(metric, metric))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_ranked_metric_by_system(metrics: pd.DataFrame, metric: str, out: Path):
    systems = ordered_system_labels(metrics)
    ncols = 2
    nrows = math.ceil(len(systems) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, max(6.2, nrows * 3.25)), constrained_layout=True)
    axes = axes.flatten()
    for ax, system in zip(axes, systems):
        grp = metrics[metrics["system_label"].eq(system)].sort_values(metric, ascending=True)
        colors = [MODEL_COLORS.get(name, "#777777") for name in grp["model_name"]]
        ax.barh(grp["model_name"], grp[metric], color=colors, height=0.72)
        ax.set_xlim(0, 1.05)
        ax.set_title(system)
        ax.set_xlabel(METRIC_TITLES.get(metric, metric))
        for idx, value in enumerate(grp[metric]):
            ax.text(min(float(value) + 0.015, 1.01), idx, f"{value:.3f}", va="center", fontsize=8)
    for ax in axes[len(systems) :]:
        ax.axis("off")
    fig.suptitle(f"Ranked Deployable Detector {METRIC_TITLES.get(metric, metric)} by System", y=1.01)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_error_tradeoff(metrics: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
    markers = {"qgrid": "o", "ruan": "s"}
    for model_name, grp in metrics.groupby("model_name", sort=False):
        color = MODEL_COLORS.get(model_name, "#777777")
        for dataset, sub in grp.groupby("dataset", sort=False):
            ax.scatter(
                sub["fpr"],
                sub["fnr"],
                s=70 + 130 * sub["f1"],
                marker=markers.get(dataset, "o"),
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.88,
                label=f"{model_name} ({DATASET_NAMES.get(dataset, dataset)})",
            )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=7, ncol=2, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("False negative rate")
    ax.set_title("Deployable Detector Error Tradeoff")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_latency(metrics: pd.DataFrame, out: Path):
    grp = metrics.sort_values("latency_ms_per_sample", ascending=True)
    colors = [MODEL_COLORS.get(name, "#777777") for name in grp["model_name"]]
    labels = grp["system_label"] + " / " + grp["model_name"]
    fig, ax = plt.subplots(figsize=(10.5, max(5.2, 0.34 * len(grp))), constrained_layout=True)
    ax.barh(labels, grp["latency_ms_per_sample"], color=colors, height=0.72)
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms/sample, log scale)")
    ax.set_title("Recorded Inference Latency by Detector")
    for idx, value in enumerate(grp["latency_ms_per_sample"]):
        ax.text(float(value) * 1.08, idx, f"{value:.4f}", va="center", fontsize=7)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_paper_figures(metrics: pd.DataFrame, out_dir: Path):
    paper_dir = out_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    deployable = deployable_only(metrics)
    for metric in PAPER_SCORE_METRICS:
        plot_metric_heatmap(
            deployable,
            metric,
            paper_dir / f"{metric}_heatmap_deployable.png",
            f"Deployable Detector {METRIC_TITLES.get(metric, metric)}",
        )
        plot_metric_heatmap(
            metrics,
            metric,
            paper_dir / f"{metric}_heatmap_with_oracle.png",
            f"Detector {METRIC_TITLES.get(metric, metric)} Including Oracle Ablations",
        )
    plot_ranked_metric_by_system(deployable, "f1", paper_dir / "f1_ranked_by_system_deployable.png")
    plot_error_tradeoff(deployable, paper_dir / "fpr_fnr_tradeoff_deployable.png")
    plot_latency(deployable, paper_dir / "latency_deployable.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/detectors")
    ap.add_argument("--out-dir", default="runs/plots")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    metrics = load_metrics(root)
    hist = load_histories(root)
    metrics.to_csv(out_dir / "detector_metrics_summary.csv", index=False)
    plot_bars(metrics, out_dir / "detector_metrics_bars.png")
    plot_individual_metric_bars(metrics, out_dir / "metrics")
    plot_learning(hist, out_dir / "detector_learning_curves.png")
    plot_individual_learning_curves(hist, out_dir / "learning_curves")
    plot_grouped_views(metrics, hist, out_dir)
    plot_paper_figures(metrics, out_dir)
    print(metrics[["system_label", "model_name", "f1", "auroc", "auprc", "fpr", "fnr", "latency_ms_per_sample"]])
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
