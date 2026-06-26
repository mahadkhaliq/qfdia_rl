#!/usr/bin/env python3
"""Plot IBM hardware smoke verification results for the Q-NPG policies."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def set_plot_style() -> None:
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


def load_hardware_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    hw = df[(df["status"] == "complete") & (df["device"] == "ibm")].copy()
    if hw.empty:
        raise ValueError(f"no completed IBM hardware rows found in {csv_path}")
    hw["bus"] = hw["bus"].astype(int)
    hw = hw.sort_values("bus")
    hw["bus_label"] = hw["bus"].astype(str) + "-bus"
    return hw


def annotate_bars(ax, bars, values, fmt="{:.3f}", dy=0.002) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            fmt.format(float(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_hardware_summary(hw: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    x = range(len(hw))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)

    sim_bars = axes[0].bar(
        [i - width / 2 for i in x],
        hw["simulator_sds"],
        width=width,
        label="Simulator",
        color="#3b6ea8",
    )
    ibm_bars = axes[0].bar(
        [i + width / 2 for i in x],
        hw["device_sds"],
        width=width,
        label="IBM hardware",
        color="#d28b39",
    )
    axes[0].set_title("Attack Impact Preserved on IBM Hardware")
    axes[0].set_ylabel("SDS")
    axes[0].set_xticks(list(x), hw["bus_label"])
    axes[0].set_ylim(0, max(hw["simulator_sds"].max(), hw["device_sds"].max()) * 1.28)
    axes[0].legend(loc="upper right")
    annotate_bars(axes[0], sim_bars, hw["simulator_sds"])
    annotate_bars(axes[0], ibm_bars, hw["device_sds"])

    stealth_loss = (1.0 - hw["device_stealth"]) * 1000.0
    delta_scaled = hw["mean_abs_attack_delta"] * 1000.0
    stealth_bars = axes[1].bar(
        [i - width / 2 for i in x],
        stealth_loss,
        width=width,
        label="1 - stealth (x1000)",
        color="#4c8f6a",
    )
    delta_bars = axes[1].bar(
        [i + width / 2 for i in x],
        delta_scaled,
        width=width,
        label="mean |a_sim-a_hw| (x1000)",
        color="#8b6bb1",
    )
    axes[1].set_title("Hardware Agreement and BDD Evasion")
    axes[1].set_ylabel("scaled value")
    axes[1].set_xticks(list(x), hw["bus_label"])
    axes[1].set_ylim(0, max(stealth_loss.max(), delta_scaled.max()) * 1.35)
    axes[1].legend(loc="upper right")
    annotate_bars(axes[1], stealth_bars, stealth_loss, fmt="{:.2f}", dy=0.05)
    annotate_bars(axes[1], delta_bars, delta_scaled, fmt="{:.2f}", dy=0.05)

    for ax in axes:
        ax.grid(axis="y", alpha=0.55)
        ax.grid(axis="x", visible=False)

    fig.suptitle("Q-NPG VQC Smoke Verification on IBM Quantum Hardware", fontsize=13)
    fig.savefig(out, dpi=220)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="paper_tables/quantum_verification_results.csv")
    ap.add_argument("--out", default="runs/plots/paper/ibm_hardware_smoke.png")
    args = ap.parse_args()

    set_plot_style()
    hw = load_hardware_rows(Path(args.csv))
    plot_hardware_summary(hw, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
