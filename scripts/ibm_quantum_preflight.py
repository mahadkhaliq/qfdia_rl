"""
IBM Quantum preflight check for QFDIA hardware verification.

This script does not ask for or print API tokens. It checks whether the IBM
Runtime package is importable, whether saved accounts are visible to Qiskit,
and, when credentials are configured, whether hardware backends are available
for the requested qubit count.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"MISSING: {type(exc).__name__}: {exc}"


def account_summary(service_cls) -> dict:
    try:
        accounts = service_cls.saved_accounts()
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    safe_accounts = []
    if isinstance(accounts, dict):
        iterable = accounts.items()
    else:
        iterable = enumerate(accounts)
    for name, account in iterable:
        if isinstance(account, dict):
            channel = account.get("channel")
            url = account.get("url")
            instance = account.get("instance")
            private_endpoint = account.get("private_endpoint")
            verify = account.get("verify")
        else:
            channel = getattr(account, "channel", None)
            url = getattr(account, "url", None)
            instance = getattr(account, "instance", None)
            private_endpoint = getattr(account, "private_endpoint", None)
            verify = getattr(account, "verify", None)
        safe_accounts.append(
            {
                "name": str(name),
                "channel": channel,
                "url": url,
                "has_instance": bool(instance),
                "instance_prefix": str(instance)[:32] + "..." if instance else None,
                "private_endpoint": private_endpoint,
                "verify": verify,
                "token": "<hidden>",
            }
        )
    return {"available": bool(safe_accounts), "count": len(safe_accounts), "accounts": safe_accounts}


def backend_summary(service, min_qubits: int, limit: int) -> dict:
    try:
        backends = service.backends(simulator=False, operational=True, min_num_qubits=min_qubits)
    except TypeError:
        backends = [
            backend
            for backend in service.backends()
            if not getattr(backend, "simulator", False)
            and getattr(backend, "num_qubits", 0) >= min_qubits
            and getattr(getattr(backend, "status", lambda: None)(), "operational", True)
        ]
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def pending_jobs_for_sort(backend) -> int:
        try:
            status = backend.status()
            pending = getattr(status, "pending_jobs", None)
            return int(pending) if pending is not None else 10**9
        except Exception:
            return 10**9

    rows = []
    sorted_backends = sorted(backends, key=pending_jobs_for_sort)
    for backend in sorted_backends[:limit]:
        status = None
        try:
            status = backend.status()
        except Exception:
            pass
        pending = getattr(status, "pending_jobs", None) if status is not None else None
        operational = getattr(status, "operational", None) if status is not None else None
        rows.append(
            {
                "name": getattr(backend, "name", str(backend)),
                "num_qubits": int(getattr(backend, "num_qubits", 0)),
                "pending_jobs": pending,
                "operational": operational,
            }
        )
    return {"available": bool(rows), "count": len(backends), "shown": rows, "selection": "lowest_pending_jobs_first"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-qubits", type=int, default=4)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="runs/quantum_architectures/ibm_preflight.json")
    args = ap.parse_args()

    payload = {
        "timestamp_unix": time.time(),
        "min_qubits": args.min_qubits,
        "versions": {
            "qiskit": safe_version("qiskit"),
            "qiskit_ibm_runtime": safe_version("qiskit_ibm_runtime"),
            "pennylane": safe_version("pennylane"),
            "pennylane_qiskit": safe_version("pennylane_qiskit"),
        },
        "account": {},
        "backends": {},
        "ready_for_hardware": False,
        "recommended_next_command": None,
    }

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        payload["account"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        payload["account"] = account_summary(QiskitRuntimeService)
        if payload["account"].get("available"):
            try:
                service = QiskitRuntimeService()
                payload["backends"] = backend_summary(service, args.min_qubits, args.limit)
            except Exception as exc:
                payload["backends"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    backend_name = None
    for item in payload.get("backends", {}).get("shown", []):
        if item.get("operational") is not False:
            backend_name = item["name"]
            break

    if backend_name:
        payload["ready_for_hardware"] = True
        payload["recommended_next_command"] = (
            "BUS=30 DEVICE=ibm IBM_BACKEND="
            + backend_name
            + " ENV_NAME=qfdia_ibm_latest sbatch scripts/run_ibm_verification_hellbender.sbatch"
        )
    else:
        payload["recommended_next_command"] = (
            "Configure IBM Quantum credentials in qfdia_ibm_latest, then rerun "
            "python scripts/ibm_quantum_preflight.py --min-qubits 4"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
