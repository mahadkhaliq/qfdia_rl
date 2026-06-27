"""
Verify that paper-result tables, plots, and quantum architecture records agree.

This is a lightweight audit script for the ignored run artifacts plus tracked
paper summaries. It intentionally avoids ML/plotting dependencies so it can run
in the base project environment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_PAPER_PLOTS = {
    "auprc_heatmap_deployable.png",
    "auprc_heatmap_with_oracle.png",
    "auroc_heatmap_deployable.png",
    "auroc_heatmap_with_oracle.png",
    "f1_heatmap_deployable.png",
    "f1_heatmap_with_oracle.png",
    "f1_ranked_by_system_deployable.png",
    "fpr_fnr_tradeoff_deployable.png",
    "latency_deployable.png",
}
EXPECTED_30_BUS_METHOD_LEARNING = {
    "30_bus_cnn1d_learning.png",
    "30_bus_mlp_learning.png",
    "30_bus_gcn_learning.png",
    "30_bus_wgcn_learning.png",
    "30_bus_gat_learning.png",
    "30_bus_qgnn_learning.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def run_set(rows: list[dict[str, str]]) -> set[str]:
    return {row["run"] for row in rows}


def check(condition: bool, message: str, failures: list[str]):
    if not condition:
        failures.append(message)


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def verify_tables(metrics_path: Path, tables_dir: Path, failures: list[str]):
    metrics = read_csv(metrics_path)
    complete = read_csv(tables_dir / "detector_metrics_complete.csv")
    deployable = read_csv(tables_dir / "detector_metrics_deployable.csv")
    oracle = read_csv(tables_dir / "detector_metrics_oracle_ablations.csv")
    best = read_csv(tables_dir / "best_deployable_by_system.csv")

    check(len(metrics) == len(complete), f"complete table row count {len(complete)} != metrics {len(metrics)}", failures)
    check(run_set(metrics) == run_set(complete), "complete table run set differs from metrics summary", failures)

    expected_deployable = {row["run"] for row in metrics if row.get("variant") != "z_plus_abs_a"}
    expected_oracle = {row["run"] for row in metrics if row.get("variant") == "z_plus_abs_a"}
    check(run_set(deployable) == expected_deployable, "deployable table does not match non-oracle metrics rows", failures)
    check(run_set(oracle) == expected_oracle, "oracle table does not match z_plus_abs_a metrics rows", failures)

    expected_systems = {row["system_label"] for row in metrics if row.get("variant") != "z_plus_abs_a"}
    check({row["system_label"] for row in best} == expected_systems, "best-by-system table does not cover all deployable systems", failures)
    return metrics


def verify_qgnn_architectures(metrics: list[dict[str, str]], detectors_root: Path, registry: Path, failures: list[str]):
    if not registry.exists():
        failures.append(f"missing registry: {registry}")
        return
    registry_text = registry.read_text()
    qgnn_runs = [row["run"] for row in metrics if row.get("model") == "qgnn"]
    for run in qgnn_runs:
        run_dir = detectors_root / run
        arch_path = run_dir / "architecture.json"
        metrics_path = run_dir / "metrics.json"
        check(arch_path.exists(), f"missing QGNN architecture.json for {run}", failures)
        check(metrics_path.exists(), f"missing QGNN metrics.json for {run}", failures)
        check(f"`{run}`" in registry_text, f"QGNN run {run} missing from architecture registry", failures)
        if arch_path.exists():
            arch = load_json(arch_path)
            for key in ["n_qubits", "q_layers", "selected_nodes", "reduced_edges", "encoding", "readout"]:
                check(key in arch and arch[key] not in (None, "", []), f"{run} architecture missing {key}", failures)


def verify_paper_plots(plots_root: Path, failures: list[str]):
    paper_dir = plots_root / "paper"
    found = {path.name for path in paper_dir.glob("*.png")} if paper_dir.exists() else set()
    missing = sorted(EXPECTED_PAPER_PLOTS.difference(found))
    check(not missing, f"missing paper plots: {missing}", failures)

    by_bus_model_dir = plots_root / "by_bus_model"
    bus_model_found = {path.name for path in by_bus_model_dir.glob("*.png")} if by_bus_model_dir.exists() else set()
    bus_model_missing = sorted(EXPECTED_30_BUS_METHOD_LEARNING.difference(bus_model_found))
    check(
        not bus_model_missing,
        f"missing 30-bus per-method learning plots: {bus_model_missing}",
        failures,
    )


def verify_quantum_verification_table(tables_dir: Path, failures: list[str]):
    rows = read_csv(tables_dir / "quantum_verification_results.csv")
    devices = {row["device"] for row in rows}
    for device in ["sim", "aer", "aer_noisy", "ibm"]:
        check(device in devices, f"quantum verification table missing {device} row", failures)

    completed = {row["device"] for row in rows if row.get("status") == "complete"}
    for device in ["sim", "aer", "aer_noisy"]:
        check(device in completed, f"quantum verification table missing completed {device} verification", failures)

    ibm_rows = [row for row in rows if row["device"] == "ibm"]
    check(bool(ibm_rows), "quantum verification table missing IBM hardware row", failures)
    if ibm_rows:
        valid_status = {"complete", "planned_after_token_rotation"}
        check(
            ibm_rows[0].get("status") in valid_status,
            f"IBM row has unexpected status {ibm_rows[0].get('status')}",
            failures,
        )


def verify_qnpg_seed_sweep_tables(tables_dir: Path, failures: list[str]):
    raw_path = tables_dir / "qnpg_seed_sweep_raw.csv"
    summary_path = tables_dir / "qnpg_seed_sweep_summary.csv"
    if not raw_path.exists() and not summary_path.exists():
        return
    raw = read_csv(raw_path)
    summary = read_csv(summary_path)
    expected_buses = {"30", "57", "118"}
    expected_methods = {
        "Q-NPG-FDIA",
        "liu_stealthy",
        "step",
        "random",
        "multiplicative",
        "coordinated_sparse",
    }
    check(len(raw) == 54, f"Q-NPG seed-sweep raw rows {len(raw)} != 54", failures)
    check(len(summary) == 18, f"Q-NPG seed-sweep summary rows {len(summary)} != 18", failures)
    check({row["bus"] for row in raw} == expected_buses, "Q-NPG seed-sweep raw table missing bus coverage", failures)
    check({row["bus"] for row in summary} == expected_buses, "Q-NPG seed-sweep summary table missing bus coverage", failures)
    check({row["method"] for row in summary} == expected_methods, "Q-NPG seed-sweep summary table missing methods", failures)
    for bus in expected_buses:
        bus_raw = [row for row in raw if row["bus"] == bus]
        bus_summary = [row for row in summary if row["bus"] == bus]
        check(len(bus_raw) == 18, f"Q-NPG seed-sweep raw table bus {bus} rows {len(bus_raw)} != 18", failures)
        check(len(bus_summary) == 6, f"Q-NPG seed-sweep summary table bus {bus} rows {len(bus_summary)} != 6", failures)
        for row in bus_summary:
            check(row.get("n_seeds") == "3", f"Q-NPG seed-sweep bus {bus} {row.get('method')} n_seeds != 3", failures)
            check(row.get("eval_n_per_seed") == "256", f"Q-NPG seed-sweep bus {bus} {row.get('method')} eval_n_per_seed != 256", failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/plots/detector_metrics_summary.csv")
    ap.add_argument("--tables-dir", default="paper_tables")
    ap.add_argument("--detectors-root", default="runs/detectors")
    ap.add_argument("--plots-root", default="runs/plots")
    ap.add_argument("--registry", default="QUANTUM_ARCHITECTURE_REGISTRY.md")
    args = ap.parse_args()

    failures: list[str] = []
    metrics = verify_tables(Path(args.metrics), Path(args.tables_dir), failures)
    verify_qgnn_architectures(metrics, Path(args.detectors_root), Path(args.registry), failures)
    verify_paper_plots(Path(args.plots_root), failures)
    verify_quantum_verification_table(Path(args.tables_dir), failures)
    verify_qnpg_seed_sweep_tables(Path(args.tables_dir), failures)

    if failures:
        print("results consistency check FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    qgnn_count = sum(1 for row in metrics if row.get("model") == "qgnn")
    print(
        "results consistency check passed: "
        f"{len(metrics)} detector rows, {qgnn_count} QGNN architecture-backed rows, "
        f"{len(EXPECTED_PAPER_PLOTS)} paper plots, {len(EXPECTED_30_BUS_METHOD_LEARNING)} 30-bus method plots, "
        "quantum verification table, Q-NPG seed-sweep tables"
    )


if __name__ == "__main__":
    main()
