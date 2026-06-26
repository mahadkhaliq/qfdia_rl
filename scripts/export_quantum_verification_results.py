#!/usr/bin/env python3
"""
Export paper-ready Q-NPG / IBM verification result tables.

This script reads verification JSON files produced by verify_ibm.py and the
token-free IBM manifest, then writes a compact markdown summary plus a CSV table.
If the real IBM hardware run has not been submitted yet, it includes a planned
hardware row so the remaining gate is explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_COLUMNS = [
    "status",
    "device",
    "bus",
    "backend",
    "policy",
    "n_qubits",
    "vqc_layers",
    "logical_depth",
    "logical_num_gates",
    "shots",
    "n_points",
    "faithfulness_max_abs",
    "simulator_stealth",
    "simulator_sds",
    "simulator_flagged_rate",
    "device_stealth",
    "device_sds",
    "device_flagged_rate",
    "mean_abs_attack_delta",
    "verdict",
    "json_path",
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def row_from_verification(path: Path) -> dict:
    payload = load_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    simulator = payload.get("simulator", {})
    device_result = payload.get("device_result", {})
    return {
        "status": "complete",
        "device": payload.get("device"),
        "bus": payload.get("bus"),
        "backend": payload.get("ibm_backend") or payload.get("fake_backend") or "",
        "policy": payload.get("policy"),
        "n_qubits": payload.get("n_qubits"),
        "vqc_layers": payload.get("vqc_layers"),
        "logical_depth": payload.get("logical_depth"),
        "logical_num_gates": payload.get("logical_num_gates"),
        "shots": payload.get("shots"),
        "n_points": payload.get("n_points"),
        "faithfulness_max_abs": payload.get("faithfulness_max_abs"),
        "simulator_stealth": simulator.get("stealth"),
        "simulator_sds": simulator.get("sds"),
        "simulator_flagged_rate": simulator.get("flagged_rate"),
        "device_stealth": device_result.get("stealth"),
        "device_sds": device_result.get("sds"),
        "device_flagged_rate": device_result.get("flagged_rate"),
        "mean_abs_attack_delta": device_result.get("mean_abs_attack_delta"),
        "verdict": payload.get("verdict"),
        "json_path": str(path),
    }


def verification_sort_key(row: dict) -> tuple:
    order = {"sim": 0, "aer": 1, "aer_noisy": 2, "ibm": 3}
    return (int(row.get("bus") or 999), order.get(str(row.get("device")), 99), str(row.get("json_path")))


def collect_completed_rows(root: Path, bus: int | None) -> list[dict]:
    rows = []
    for path in sorted(root.glob("verify_*.json")):
        if path.name.startswith("verify_results") or path.name.startswith("ibm_verification_manifest"):
            continue
        payload = load_json(path)
        if not payload or "device_result" not in payload:
            continue
        if bus is not None and int(payload.get("bus", -1)) != bus:
            continue
        rows.append(row_from_verification(path))
    return sorted(deduplicate_latest_aliases(rows), key=verification_sort_key)


def deduplicate_latest_aliases(rows: list[dict]) -> list[dict]:
    """Prefer tagged verification JSONs over latest aliases for identical runs."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("device"),
            str(row.get("bus")),
            row.get("backend") or "",
            row.get("shots"),
            row.get("n_points"),
            row.get("verdict") or "",
            row.get("device_stealth"),
            row.get("device_sds"),
            row.get("device_flagged_rate"),
            row.get("mean_abs_attack_delta"),
        )
        groups.setdefault(key, []).append(row)

    selected = []
    for group in groups.values():
        if len(group) == 1:
            selected.append(group[0])
            continue
        tagged = [
            row for row in group
            if not Path(str(row.get("json_path", ""))).stem.endswith(f"_{row.get('bus')}")
        ]
        selected.append(sorted(tagged or group, key=lambda r: str(r.get("json_path")))[-1])
    return selected


def baseline_circuit_row(rows: list[dict], bus: int) -> dict:
    for device in ["sim", "aer", "aer_noisy"]:
        for row in rows:
            if row.get("device") == device and int(row.get("bus") or -1) == bus:
                return row
    return {}


def default_planned_hardware_run(args) -> dict:
    if args.bus is None:
        return {}
    return {
        "bus": args.bus,
        "ibm_backend": args.backend,
        "policy": args.policy or f"runs/policies/qnpg_{args.bus}_policy.npz",
        "shots": args.shots,
        "n_points": args.n_points,
        "result_tag": args.result_tag,
        "expected_json": f"runs/quantum_architectures/verify_ibm_{args.bus}_{args.result_tag}.json",
    }


def planned_ibm_row(manifest_path: Path, rows: list[dict], default_plan: dict) -> dict | None:
    manifest = load_json(manifest_path)
    planned = default_plan.copy()
    if manifest:
        planned.update({k: v for k, v in manifest.get("planned_hardware_run", {}).items() if v not in (None, "")})
    if not planned.get("bus"):
        return None
    bus = planned.get("bus")
    expected_json = planned.get("expected_json")
    has_complete_ibm = any(
        row.get("device") == "ibm"
        and int(row.get("bus") or -1) == int(bus or -1)
        and (not expected_json or row.get("json_path") == expected_json)
        for row in rows
    )
    if has_complete_ibm:
        return None
    baseline = baseline_circuit_row(rows, int(bus))
    return {
        "status": "planned_after_token_rotation",
        "device": "ibm",
        "bus": bus,
        "backend": planned.get("ibm_backend"),
        "policy": planned.get("policy"),
        "n_qubits": baseline.get("n_qubits", ""),
        "vqc_layers": baseline.get("vqc_layers", ""),
        "logical_depth": baseline.get("logical_depth", ""),
        "logical_num_gates": baseline.get("logical_num_gates", ""),
        "shots": planned.get("shots"),
        "n_points": planned.get("n_points"),
        "faithfulness_max_abs": "",
        "simulator_stealth": "",
        "simulator_sds": "",
        "simulator_flagged_rate": "",
        "device_stealth": "",
        "device_sds": "",
        "device_flagged_rate": "",
        "mean_abs_attack_delta": "",
        "verdict": "pending_token_rotation_and_qpu_submission",
        "json_path": expected_json,
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def write_markdown(path: Path, rows: list[dict], manifest_path: Path):
    with open(path, "w") as handle:
        handle.write("# Quantum Verification Results\n\n")
        handle.write("Regenerate this file with:\n\n")
        handle.write("```bash\npython scripts/export_quantum_verification_results.py\n```\n\n")
        handle.write("## Q-NPG Policy Verification\n\n")
        handle.write("| Status | Device | Backend | Bus | Qubits | Points | Shots | Device stealth | Device SDS | Flagged | Attack delta | Verdict | JSON |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row.get('status', '-')} | {row.get('device', '-')} | {row.get('backend') or '-'} | "
                f"{fmt(row.get('bus'), 0)} | {fmt(row.get('n_qubits'), 0)} | {fmt(row.get('n_points'), 0)} | "
                f"{fmt(row.get('shots'), 0)} | {fmt(row.get('device_stealth'))} | {fmt(row.get('device_sds'))} | "
                f"{fmt(row.get('device_flagged_rate'))} | {fmt(row.get('mean_abs_attack_delta'))} | "
                f"{row.get('verdict') or '-'} | `{row.get('json_path') or '-'}` |\n"
            )
        handle.write("\n## Notes\n\n")
        handle.write("- `sim`, `aer`, and `aer_noisy` rows are completed non-hardware checks of the trained Q-NPG VQC actor.\n")
        ibm_complete_count = sum(1 for row in rows if row.get("device") == "ibm" and row.get("status") == "complete")
        if ibm_complete_count > 1:
            handle.write("- The `ibm` rows are completed real-hardware smoke verifications of the trained Q-NPG VQC actor.\n")
        elif ibm_complete_count == 1:
            handle.write("- The `ibm` row is a completed real-hardware smoke verification of the trained Q-NPG VQC actor.\n")
        else:
            handle.write("- The `ibm` row remains planned until the IBM API key has been rotated/reconfirmed and the hardware smoke job has completed.\n")
        handle.write("- The hardware smoke run is intentionally small: one circuit evaluation per operating point, with the listed shot count.\n")
        handle.write(f"- Hardware safety gate and exact command are recorded in `{manifest_path}`.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/quantum_architectures")
    ap.add_argument("--bus", type=int, default=None, help="Filter to one bus; default exports all completed buses.")
    ap.add_argument("--manifest", default="runs/quantum_architectures/ibm_verification_manifest.json")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--n-points", type=int, default=4)
    ap.add_argument("--result-tag", default="ibm_smoke_30")
    ap.add_argument("--csv-out", default="paper_tables/quantum_verification_results.csv")
    ap.add_argument("--md-out", default="QUANTUM_VERIFICATION_RESULTS.md")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect_completed_rows(root, args.bus)
    planned = planned_ibm_row(Path(args.manifest), rows, default_planned_hardware_run(args))
    if planned:
        rows.append(planned)
    rows = sorted(rows, key=verification_sort_key)
    write_csv(Path(args.csv_out), rows)
    write_markdown(Path(args.md_out), rows, Path(args.manifest))
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
