"""
plot_curves.py
==============
Build the Q-NPG-FDIA learning-curve figure from the saved history CSVs.

Reads outputs/qnpg_<bus>_history.csv for bus in {30, 57, 118} (whichever exist)
and writes outputs/learning_curves.png : three columns (SDS / stealth / ASR)
vs update, one line per bus size.

Usage:
    python plot_curves.py                      # looks in ./outputs
    python plot_curves.py --dir outputs_run1   # a different results dir
    python plot_curves.py --buses 30 57 118
"""
from __future__ import annotations
import os, csv, argparse
import matplotlib
matplotlib.use("Agg")                 # headless (HPC login node, no display)
import matplotlib.pyplot as plt


def load_history(path):
    rows = {"update": [], "state_dev": [], "stealth": [], "asr": [], "ep_return": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in rows:
                if k in r and r[k] != "":
                    rows[k].append(float(r[k]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs")
    ap.add_argument("--buses", nargs="+", type=int, default=[30, 57, 118])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    colors = {30: "#2563eb", 57: "#d97706", 118: "#dc2626"}
    series = []
    for b in args.buses:
        p = os.path.join(args.dir, f"qnpg_{b}_history.csv")
        if os.path.exists(p):
            series.append((b, load_history(p)))
        else:
            print(f"  (skipping bus {b}: {p} not found)")
    if not series:
        print("No history CSVs found. Run training first."); return

    panels = [("state_dev", "State-estimate deviation (impact)", "mean SDS"),
              ("stealth",   "Stealth margin (1 = deep below BDD threshold)", "stealth score"),
              ("asr",       "Attack success rate", "ASR")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title, ylab) in zip(axes, panels):
        for b, h in series:
            ax.plot(h["update"], h[key], color=colors.get(b, None),
                    lw=2, label=f"IEEE {b}-bus")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training update"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Q-NPG-FDIA learning curves across IEEE systems", fontsize=13, y=1.02)
    fig.tight_layout()

    out = args.out or os.path.join(args.dir, "learning_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved figure -> {out}")


if __name__ == "__main__":
    main()
