"""
Check the local quantum software stack for IBM verification readiness.

This is intentionally lightweight: it imports the relevant packages, prints
versions, and performs a tiny PennyLane-Qiskit Aer circuit evaluation.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path


def version_of(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    return getattr(mod, "__version__", "unknown")


def main():
    modules = [
        "pennylane",
        "pennylane_qiskit",
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
        "numpy",
        "scipy",
        "pandas",
        "pandapower",
        "pyarrow",
    ]
    versions = {}
    for name in modules:
        try:
            versions[name] = version_of(name)
            print(f"{name}: {versions[name]}")
        except Exception as exc:
            versions[name] = f"MISSING: {type(exc).__name__}: {exc}"
            print(f"{name}: {versions[name]}")

    if not str(versions.get("pennylane_qiskit", "")).startswith("MISSING"):
        import pennylane as qml
        import numpy as np

        dev = qml.device("qiskit.aer", wires=2, shots=256)

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        out = np.asarray(circuit(0.123), dtype=float)
        print(f"qiskit.aer smoke output: {out.tolist()}")
        versions["qiskit_aer_smoke_output"] = out.tolist()
    else:
        versions["qiskit_aer_smoke_output"] = "SKIPPED: pennylane_qiskit missing"
        print("qiskit.aer smoke output: SKIPPED")

    out_dir = Path("runs/quantum_architectures")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "quantum_stack_versions.json", "w") as f:
        json.dump(versions, f, indent=2)
    print(f"wrote {out_dir / 'quantum_stack_versions.json'}")


if __name__ == "__main__":
    main()
