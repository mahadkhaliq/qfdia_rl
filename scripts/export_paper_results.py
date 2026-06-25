"""
Export paper-ready FDIA detector result tables from the aggregated metrics CSV.

Input:
  runs/plots/detector_metrics_summary.csv

Output:
  PAPER_RESULTS.md

The markdown file is intentionally tracked because it is the compact narrative
artifact we will cite while writing the paper. Large plots and model artifacts
remain under ignored run directories.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Union


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "fpr",
    "fnr",
    "auroc",
    "auprc",
    "latency_ms_per_sample",
]
TABLE_COLUMNS = [
    "run",
    "dataset",
    "bus",
    "model",
    "variant",
    "system_label",
    "model_name",
    *METRIC_COLUMNS,
    "n",
    "threshold",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def fmt(value: Union[str, float], digits: int = 4) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, str):
        value = float(value)
    if value != value:
        return "-"
    return f"{value:.{digits}f}"


def fmt_sci(value: Union[str, float], digits: int = 2) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, str):
        value = float(value)
    if value != value:
        return "-"
    return f"{value:.{digits}e}"


def md(value: str) -> str:
    return value.replace("|", "\\|")


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda r: (
            int(float(r.get("dataset_order") or 99)),
            int(float(r.get("bus") or 999)),
            int(float(r.get("model_order") or 99)),
            int(float(r.get("variant_order") or 99)),
            r.get("run", ""),
        ),
    )


def table_header(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |\n" + "| " + " | ".join(["---"] * len(columns)) + " |\n"


def write_overview_table(f, rows: list[dict[str, str]]):
    f.write("## Overall Detector Metrics\n\n")
    columns = ["Dataset/System", "Model", "F1", "AUROC", "AUPRC", "FPR", "FNR", "Latency ms/sample"]
    f.write(table_header(columns))
    for row in sort_rows(rows):
        f.write(
            f"| {md(row['system_label'])} | {md(row['model_name'])} | {fmt(row['f1'])} | {fmt(row['auroc'])} | "
            f"{fmt(row['auprc'])} | {fmt(row['fpr'])} | {fmt(row['fnr'])} | {fmt(row['latency_ms_per_sample'])} |\n"
        )
    f.write("\n")


def is_oracle_row(row: dict[str, str]) -> bool:
    return row.get("variant") == "z_plus_abs_a"


def write_best_by_system(f, rows: list[dict[str, str]]):
    f.write("## Best Detector Per Dataset/System\n\n")
    columns = ["Dataset/System", "Best Model", "F1", "AUROC", "AUPRC", "FPR", "FNR"]
    f.write(table_header(columns))
    by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_system[row["system_label"]].append(row)
    for system in sorted(by_system):
        best = max(by_system[system], key=lambda r: as_float(r, "f1"))
        f.write(
            f"| {md(system)} | {md(best['model_name'])} | {fmt(best['f1'])} | {fmt(best['auroc'])} | "
            f"{fmt(best['auprc'])} | {fmt(best['fpr'])} | {fmt(best['fnr'])} |\n"
        )
    f.write("\n")


def write_best_deployable_by_system(f, rows: list[dict[str, str]]):
    deployable = [row for row in rows if not is_oracle_row(row)]
    f.write("## Best Deployable Detector Per Dataset/System\n\n")
    columns = ["Dataset/System", "Best Model", "F1", "AUROC", "AUPRC", "FPR", "FNR"]
    f.write(table_header(columns))
    by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deployable:
        by_system[row["system_label"]].append(row)
    for system in sorted(by_system):
        best = max(by_system[system], key=lambda r: as_float(r, "f1"))
        f.write(
            f"| {md(system)} | {md(best['model_name'])} | {fmt(best['f1'])} | {fmt(best['auroc'])} | "
            f"{fmt(best['auprc'])} | {fmt(best['fpr'])} | {fmt(best['fnr'])} |\n"
        )
    f.write("\n")


def write_by_bus_tables(f, rows: list[dict[str, str]]):
    f.write("## Metrics By Bus Number\n\n")
    by_bus: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bus[row["bus"]].append(row)
    for bus in sorted(by_bus, key=lambda x: int(float(x))):
        f.write(f"### IEEE {bus}-Bus\n\n")
        columns = ["Dataset/System", "Model", "F1", "Balanced Acc.", "Precision", "Recall", "MCC", "AUROC", "AUPRC"]
        f.write(table_header(columns))
        for row in sort_rows(by_bus[bus]):
            f.write(
                f"| {md(row['system_label'])} | {md(row['model_name'])} | {fmt(row['f1'])} | "
                f"{fmt(row['balanced_accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
                f"{fmt(row['mcc'])} | {fmt(row['auroc'])} | {fmt(row['auprc'])} |\n"
            )
        f.write("\n")


def write_figure_index(f, plots_root: Path):
    f.write("## Figure Index\n\n")
    f.write("### Combined Figures\n\n")
    for rel in ["detector_metrics_bars.png", "detector_learning_curves.png"]:
        path = plots_root / rel
        f.write(f"- `{path}`\n")
    f.write("\n### Paper Figures\n\n")
    for path in sorted((plots_root / "paper").glob("*.png")):
        f.write(f"- `{path}`\n")
    f.write("\n### Per-Bus Figures\n\n")
    for path in sorted((plots_root / "by_bus").glob("*.png")):
        f.write(f"- `{path}`\n")
    f.write("\n### Per-Bus / Per-Method Figures\n\n")
    for path in sorted((plots_root / "by_bus_model").glob("*.png")):
        f.write(f"- `{path}`\n")
    f.write("\n### Per-System Figures\n\n")
    for path in sorted((plots_root / "by_system").glob("*.png")):
        f.write(f"- `{path}`\n")
    f.write("\n")


def project_row(row: dict[str, str]) -> dict[str, str]:
    return {col: row.get(col, "") for col in TABLE_COLUMNS}


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str] = TABLE_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def export_csv_tables(rows: list[dict[str, str]], tables_dir: Path):
    sorted_all = [project_row(row) for row in sort_rows(rows)]
    deployable = [row for row in rows if not is_oracle_row(row)]
    oracle = [row for row in rows if is_oracle_row(row)]
    write_csv(tables_dir / "detector_metrics_complete.csv", sorted_all)
    write_csv(tables_dir / "detector_metrics_deployable.csv", [project_row(row) for row in sort_rows(deployable)])
    write_csv(tables_dir / "detector_metrics_oracle_ablations.csv", [project_row(row) for row in sort_rows(oracle)])

    best_rows = []
    by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deployable:
        by_system[row["system_label"]].append(row)
    for system in sorted(by_system):
        best_rows.append(project_row(max(by_system[system], key=lambda r: as_float(r, "f1"))))
    write_csv(tables_dir / "best_deployable_by_system.csv", best_rows)

    by_bus: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bus[row["bus"]].append(row)
    for bus, bus_rows in by_bus.items():
        write_csv(tables_dir / f"detector_metrics_{int(float(bus))}_bus.csv", [project_row(row) for row in sort_rows(bus_rows)])


def write_table_index(f, tables_dir: Path):
    f.write("## Complete Table Files\n\n")
    for path in sorted(tables_dir.glob("*.csv")):
        f.write(f"- `{path}`\n")
    f.write("\n")


def write_qgnn_threshold_sweeps(f, detectors_root: Path):
    summaries = sorted(detectors_root.glob("*/threshold_sweep_summary.json"))
    if not summaries:
        return
    f.write("## QGNN Threshold Sweep Summaries\n\n")
    f.write("These rows report post-hoc threshold choices on saved QGNN test probabilities for recall/FPR tradeoff analysis.\n\n")
    columns = ["Run", "Criterion", "Threshold", "F1", "Balanced Acc.", "Precision", "Recall", "FPR", "FNR", "MCC"]
    f.write(table_header(columns))
    keys = [
        ("best_f1", "Best F1"),
        ("best_balanced_accuracy", "Best Balanced Acc."),
        ("best_mcc", "Best MCC"),
        ("best_recall_at_fpr_0.05", "Best Recall @ FPR <= 0.05"),
        ("best_recall_at_fpr_0.10", "Best Recall @ FPR <= 0.10"),
        ("best_recall_at_fpr_0.20", "Best Recall @ FPR <= 0.20"),
    ]
    for path in summaries:
        with open(path) as handle:
            summary = json.load(handle)
        run_name = Path(summary["run_dir"]).name
        for key, label in keys:
            row = summary.get(key)
            if not row:
                continue
            f.write(
                f"| `{md(run_name)}` | {label} | {fmt(row['threshold'])} | {fmt(row['f1'])} | "
                f"{fmt(row['balanced_accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
                f"{fmt(row['fpr'])} | {fmt(row['fnr'])} | {fmt(row['mcc'])} |\n"
            )
    f.write("\n")


def write_sds_ceiling_table(f, tables_dir: Path):
    path = tables_dir / "sds_ceiling_ratios.csv"
    if not path.exists():
        return
    rows = read_rows(path)
    if not rows:
        return
    f.write("## Q-NPG Analytical Stealth Ceiling\n\n")
    f.write("These rows compare the learned Q-NPG-FDIA mean SDS against a direct max-SDS optimizer under the same linearized stealth constraint `a = Hc` and per-meter box bound.\n\n")
    columns = ["Bus", "a_max", "SDS Ceiling", "Learned SDS", "Learned/Ceiling", "Stealth Residual", "BDD Tau"]
    f.write(table_header(columns))
    for row in sorted(rows, key=lambda r: int(float(r["bus"]))):
        f.write(
            f"| {int(float(row['bus']))} | {fmt(row['a_max'])} | {fmt(row['SDS_ceiling'])} | "
            f"{fmt(row['SDS_learned'])} | {fmt(row['learned_over_ceiling'])} | "
            f"{fmt_sci(row['stealth_residual'])} | {fmt(row['tau_bdd'], digits=3)} |\n"
        )
    f.write("\n")


def write_metric_definitions(f):
    f.write("## Metric Definitions For Paper\n\n")
    f.write("- `+|a|` model labels denote QGrid-only oracle/residual-aware ablations using the synthetic attack vector magnitude.\n")
    f.write("- F1: harmonic mean of precision and recall for attack detection.\n")
    f.write("- AUROC: threshold-independent separability between normal and attack samples.\n")
    f.write("- AUPRC: precision-recall area, useful when attack/normal ratios change.\n")
    f.write("- FPR: normal samples incorrectly flagged as attacks.\n")
    f.write("- FNR: attacks missed by the detector.\n")
    f.write("- MCC: balanced correlation-style score over all confusion-matrix cells.\n")
    f.write("- Latency: measured inference time per sample in the recorded run environment.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/plots/detector_metrics_summary.csv")
    ap.add_argument("--plots-root", default="runs/plots")
    ap.add_argument("--detectors-root", default="runs/detectors")
    ap.add_argument("--out", default="PAPER_RESULTS.md")
    ap.add_argument("--tables-dir", default="paper_tables")
    args = ap.parse_args()

    rows = read_rows(Path(args.metrics))
    plots_root = Path(args.plots_root)
    tables_dir = Path(args.tables_dir)
    export_csv_tables(rows, tables_dir)
    with open(args.out, "w") as f:
        f.write("# Paper Results: FDIA Detector Comparison\n\n")
        f.write("Regenerate this file after new detector runs with:\n\n")
        f.write("```bash\npython scripts/export_paper_results.py\n```\n\n")
        write_best_deployable_by_system(f, rows)
        write_best_by_system(f, rows)
        write_by_bus_tables(f, rows)
        write_overview_table(f, rows)
        write_qgnn_threshold_sweeps(f, Path(args.detectors_root))
        write_sds_ceiling_table(f, tables_dir)
        write_table_index(f, tables_dir)
        write_figure_index(f, plots_root)
        write_metric_definitions(f)
    print(f"wrote {args.out} and CSV tables under {tables_dir} from {args.metrics}")


if __name__ == "__main__":
    main()
