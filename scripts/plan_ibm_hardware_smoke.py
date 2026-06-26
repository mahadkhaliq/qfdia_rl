#!/usr/bin/env python3
"""
Create a small, token-free IBM hardware smoke-test plan.

This is a planning/reporting helper only. It does not contact IBM Quantum and
does not run circuits. The goal is to make the first hardware submission
explicit: qubits, shots, operating points, approximate circuit calls, expected
output JSON, and the Slurm command to use from Hellbender.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_quantum_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def planned_backend(manifest: dict, fallback: str) -> str:
    run = manifest.get("planned_hardware_run", {})
    return run.get("ibm_backend") or fallback


def existing_status(rows: list[dict[str, str]]) -> dict[str, str]:
    status = {}
    for row in rows:
        device = row.get("device", "")
        if device:
            status[device] = row.get("status", "")
    return status


def make_plan(args) -> dict:
    manifest = load_json(Path(args.manifest))
    qrows = read_quantum_table(Path(args.quantum_table))
    backend = args.backend or planned_backend(manifest, args.default_backend)
    result_tag = args.result_tag
    expected_json = f"runs/quantum_architectures/verify_ibm_{args.bus}_{result_tag}.json"
    latest_json = f"runs/quantum_architectures/verify_ibm_{args.bus}.json"
    slurm_command = (
        f"BUS={args.bus} DEVICE=ibm IBM_BACKEND={backend} ENV_NAME={args.env_name} "
        f"SHOTS={args.shots} N_POINTS={args.n_points} POLICY={args.policy} "
        f"RESULT_TAG={result_tag} sbatch scripts/run_ibm_verification_hellbender.sbatch"
    )
    circuit_evaluations = args.n_points
    total_shots = circuit_evaluations * args.shots
    return {
        "bus": args.bus,
        "n_qubits": args.n_qubits,
        "backend": backend,
        "shots_per_circuit": args.shots,
        "operating_points": args.n_points,
        "estimated_hardware_circuit_evaluations": circuit_evaluations,
        "estimated_total_shots": total_shots,
        "policy": args.policy,
        "result_tag": result_tag,
        "expected_json": expected_json,
        "latest_json": latest_json,
        "slurm_command": slurm_command,
        "existing_verification_status": existing_status(qrows),
        "safety_gate": [
            "Rotate/recreate the IBM API token if the old exposed token has not been rotated.",
            "Run scripts/ibm_quantum_preflight.py from Slurm/OOD and confirm token is redacted as <hidden>.",
            "Submit only this small first run before increasing shots or operating points.",
            "Archive stdout/stderr and the resulting verify_ibm JSON with paper artifacts.",
        ],
    }


def write_markdown(path: Path, plan: dict) -> None:
    with path.open("w") as handle:
        handle.write("# IBM Hardware Smoke-Test Plan\n\n")
        handle.write("This is a token-free plan for the first real-QPU verification of the Q-NPG VQC actor.\n\n")
        handle.write("## Scope\n\n")
        handle.write(f"- Bus: IEEE {plan['bus']}-bus\n")
        handle.write(f"- Qubits: {plan['n_qubits']}\n")
        handle.write(f"- Backend: `{plan['backend']}`\n")
        handle.write(f"- Operating points: {plan['operating_points']}\n")
        handle.write(f"- Shots per circuit: {plan['shots_per_circuit']}\n")
        handle.write(f"- Estimated hardware circuit evaluations: {plan['estimated_hardware_circuit_evaluations']}\n")
        handle.write(f"- Estimated total shots: {plan['estimated_total_shots']}\n")
        handle.write(f"- Policy: `{plan['policy']}`\n\n")

        handle.write("## Existing Verification Status\n\n")
        handle.write("| Device | Status |\n| --- | --- |\n")
        for device, status in sorted(plan["existing_verification_status"].items()):
            handle.write(f"| {device} | {status or '-'} |\n")
        handle.write("\n")

        handle.write("## Slurm Command\n\n")
        handle.write("Run from `/home/mkfqm/qfdia_rl_ondemand` or another clean code checkout on Hellbender:\n\n")
        handle.write("```bash\n")
        handle.write(plan["slurm_command"])
        handle.write("\n```\n\n")

        handle.write("Expected output JSON:\n\n")
        handle.write(f"- `{plan['expected_json']}`\n")
        handle.write(f"- latest alias: `{plan['latest_json']}`\n\n")

        handle.write("## Safety Gate\n\n")
        for item in plan["safety_gate"]:
            handle.write(f"- {item}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=30)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--n-points", type=int, default=4)
    ap.add_argument("--env-name", default="synthgrad")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--default-backend", default="ibm_fez")
    ap.add_argument("--policy", default="runs/policies/qnpg_30_policy.npz")
    ap.add_argument("--result-tag", default="ibm_smoke_30")
    ap.add_argument("--manifest", default="runs/quantum_architectures/ibm_verification_manifest.json")
    ap.add_argument("--quantum-table", default="paper_tables/quantum_verification_results.csv")
    ap.add_argument("--out", default="runs/quantum_architectures/ibm_hardware_smoke_plan.json")
    ap.add_argument("--md-out", default="IBM_HARDWARE_SMOKE_PLAN.md")
    args = ap.parse_args()

    plan = make_plan(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        json.dump(plan, handle, indent=2)
    write_markdown(Path(args.md_out), plan)
    print(f"wrote {out}")
    print(f"wrote {args.md_out}")
    print(plan["slurm_command"])


if __name__ == "__main__":
    main()
