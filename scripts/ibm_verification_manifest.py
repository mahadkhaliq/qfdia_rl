#!/usr/bin/env python3
"""
Create a reproducible IBM Quantum verification manifest.

The manifest is intentionally token-free. It gathers existing simulator/Aer
verification JSON, IBM preflight metadata, and the exact Slurm command planned
for a small real-hardware verification run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEFAULT_DEVICES = ["sim", "aer", "aer_noisy"]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def verification_summary(path: Path) -> dict:
    payload = load_json(path)
    if payload is None:
        return {"path": str(path), "available": False}
    device_result = payload.get("device_result", {})
    simulator = payload.get("simulator", {})
    return {
        "path": str(path),
        "available": True,
        "device": payload.get("device"),
        "bus": payload.get("bus"),
        "policy": payload.get("policy"),
        "shots": payload.get("shots"),
        "n_points": payload.get("n_points"),
        "n_qubits": payload.get("n_qubits"),
        "vqc_layers": payload.get("vqc_layers"),
        "logical_depth": payload.get("logical_depth"),
        "logical_num_gates": payload.get("logical_num_gates"),
        "faithfulness_max_abs": payload.get("faithfulness_max_abs"),
        "simulator_stealth": simulator.get("stealth"),
        "simulator_sds": simulator.get("sds"),
        "simulator_flagged_rate": simulator.get("flagged_rate"),
        "device_stealth": device_result.get("stealth"),
        "device_sds": device_result.get("sds"),
        "device_flagged_rate": device_result.get("flagged_rate"),
        "mean_abs_attack_delta": device_result.get("mean_abs_attack_delta"),
        "verdict": payload.get("verdict"),
    }


def recommended_backend(preflight: dict | None, requested_backend: str | None) -> str | None:
    if requested_backend:
        return requested_backend
    if preflight:
        for item in preflight.get("backends", {}).get("shown", []):
            if item.get("operational") is not False:
                return item.get("name")
    return None


def build_sbatch_command(args, backend: str | None, policy: str) -> str | None:
    if not backend:
        return None
    parts = [
        f"BUS={args.bus}",
        "DEVICE=ibm",
        f"IBM_BACKEND={backend}",
        f"ENV_NAME={args.env_name}",
        f"SHOTS={args.shots}",
        f"N_POINTS={args.n_points}",
        f"POLICY={policy}",
        f"RESULT_TAG={args.result_tag}",
        "sbatch scripts/run_ibm_verification_hellbender.sbatch",
    ]
    return " ".join(parts)


def write_markdown(path: Path, manifest: dict):
    rows = manifest["existing_verifications"]
    with open(path, "w") as handle:
        handle.write("# IBM Verification Manifest\n\n")
        handle.write(f"- Generated: `{manifest['generated_at_unix']}`\n")
        handle.write(f"- Bus: IEEE {manifest['planned_hardware_run']['bus']}-bus\n")
        handle.write(f"- Policy: `{manifest['planned_hardware_run']['policy']}`\n")
        handle.write(f"- Planned backend: `{manifest['planned_hardware_run']['ibm_backend'] or 'unavailable'}`\n")
        handle.write(f"- Shots / points: {manifest['planned_hardware_run']['shots']} / {manifest['planned_hardware_run']['n_points']}\n")
        handle.write(f"- Token status: {manifest['safety']['token_rotation_status']}\n\n")

        handle.write("## Existing Non-Hardware Verification\n\n")
        handle.write("| Device | Available | Qubits | Points | Shots | Device stealth | Device SDS | Flagged | Verdict | JSON |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row.get('device', '-')} | {row['available']} | {fmt(row.get('n_qubits'), 0)} | "
                f"{fmt(row.get('n_points'), 0)} | {fmt(row.get('shots'), 0)} | "
                f"{fmt(row.get('device_stealth'))} | {fmt(row.get('device_sds'))} | "
                f"{fmt(row.get('device_flagged_rate'))} | {row.get('verdict', '-')} | `{row['path']}` |\n"
            )

        handle.write("\n## Planned IBM Command\n\n")
        command = manifest["planned_hardware_run"].get("sbatch_command")
        if command:
            handle.write("```bash\n")
            handle.write(command)
            handle.write("\n```\n\n")
        else:
            handle.write("No hardware command is available because no operational backend was selected.\n\n")

        handle.write("## Safety Gate\n\n")
        for item in manifest["safety"]["required_before_submit"]:
            handle.write(f"- {item}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=30)
    ap.add_argument("--root", default="runs/quantum_architectures")
    ap.add_argument("--preflight", default="runs/quantum_architectures/ibm_preflight_synthgrad.json")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--env-name", default="qfdia_ibm_latest")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--n-points", type=int, default=4)
    ap.add_argument("--result-tag", default="ibm_smoke_30")
    ap.add_argument("--out", default="runs/quantum_architectures/ibm_verification_manifest.json")
    ap.add_argument("--md-out", default="IBM_VERIFICATION_MANIFEST.md")
    args = ap.parse_args()

    root = Path(args.root)
    policy = args.policy or f"runs/policies/qnpg_{args.bus}_policy.npz"
    preflight = load_json(Path(args.preflight))
    backend = recommended_backend(preflight, args.backend)
    existing = [verification_summary(root / f"verify_{device}_{args.bus}.json") for device in DEFAULT_DEVICES]
    command = build_sbatch_command(args, backend, policy)
    manifest = {
        "generated_at_unix": time.time(),
        "preflight_path": args.preflight,
        "preflight_ready_for_hardware": bool(preflight and preflight.get("ready_for_hardware")),
        "preflight_versions": (preflight or {}).get("versions", {}),
        "existing_verifications": existing,
        "planned_hardware_run": {
            "bus": args.bus,
            "device": "ibm",
            "ibm_backend": backend,
            "env_name": args.env_name,
            "policy": policy,
            "shots": args.shots,
            "n_points": args.n_points,
            "result_tag": args.result_tag,
            "expected_json": f"runs/quantum_architectures/verify_ibm_{args.bus}_{args.result_tag}.json",
            "sbatch_command": command,
        },
        "safety": {
            "token_rotation_status": "required before real hardware submission if the old exposed token has not been rotated",
            "required_before_submit": [
                "Rotate/recreate the IBM API key if not already done after the earlier accidental token print.",
                "Rerun scripts/ibm_quantum_preflight.py and confirm token is redacted as <hidden>.",
                "Use a small first hardware smoke run: 4 operating points and 1024 shots.",
                "Archive the Slurm stdout/stderr and resulting verify_ibm JSON with the paper artifacts.",
            ],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(manifest, handle, indent=2)
    write_markdown(Path(args.md_out), manifest)
    print(f"wrote {out}")
    print(f"wrote {args.md_out}")
    if command:
        print(command)


if __name__ == "__main__":
    main()
