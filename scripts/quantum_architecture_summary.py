"""
Write quantum circuit and architecture summaries for Q-NPG-FDIA and QGNN planning.

The Q-NPG policy architecture is already implemented in models/vqc_policy.py:
  AngleEmbedding(Y) -> StronglyEntanglingLayers -> PauliZ expectation readout.

This script records those circuit specs per bus and writes text circuit drawings
that can be used in reports and as a checklist before IBM Quantum verification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pennylane as qml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUS_CONFIGS


def qnpg_circuit(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(angles, theta):
        qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(theta, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def qnpg_summary(bus: int) -> dict:
    cfg = BUS_CONFIGS[bus]
    theta_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=cfg.vqc_layers,
        n_wires=cfg.n_qubits,
    )
    return {
        "name": f"Q-NPG-FDIA VQC actor, IEEE {bus}-bus",
        "bus": bus,
        "n_qubits": cfg.n_qubits,
        "vqc_layers": cfg.vqc_layers,
        "ansatz": "PennyLane StronglyEntanglingLayers",
        "encoding": "AngleEmbedding with Y rotations after classical tanh encoder",
        "readout": "PauliZ expectation on every qubit",
        "theta_shape": list(theta_shape),
        "theta_parameters": int(np.prod(theta_shape)),
        "classical_encoder": f"W_enc maps observation dimension m+3 to {cfg.n_qubits} angles",
        "classical_head": "W_act maps PauliZ readout to full attack vector",
        "ibm_verification": "Use verify_ibm.py for simulator/Aer/noisy-Aer/IBM inference verification",
    }


def qgnn_detector_templates() -> list[dict]:
    return [
        {
            "name": "Reduced QGNN detector, edge-entangler template",
            "purpose": "Quantum analogue of graph convolution/message passing for small reduced grids.",
            "encoding": "Per-node or pooled node features encoded with RY/RZ angle rotations.",
            "graph_operator": "Two-qubit entanglers on selected physical grid edges, e.g. CZ/RZZ/RXX.",
            "readout": "PauliZ expectations pooled into a classical binary FDIA head.",
            "first_target": "Reduced 4-8 qubit subgraph before full 30/57/118-bus scaling.",
            "verification": "Compare simulator vs noisy simulator vs IBM hardware on fixed test snapshots.",
        },
        {
            "name": "Quantum attention detector, compressed-feature template",
            "purpose": "Quantum analogue of GAT for IBM-feasible widths.",
            "encoding": "Classically pool graph features to q qubits, then use variational entangling layers.",
            "graph_operator": "Attention remains classical or hybrid; VQC supplies nonlinear quantum feature map.",
            "readout": "PauliZ expectations plus classical logistic/MLP detector head.",
            "first_target": "4, 6, and 8 qubits to match current Q-NPG bus presets.",
            "verification": "Use same metrics as CNN/MLP/GCN/GAT, plus shots/depth/backend metadata.",
        },
    ]


def write_qnpg_circuit_drawings(out_dir: Path, summaries: list[dict]):
    for item in summaries:
        bus = item["bus"]
        n_qubits = item["n_qubits"]
        layers = item["vqc_layers"]
        theta_shape = tuple(item["theta_shape"])
        circuit = qnpg_circuit(n_qubits)
        angles = np.zeros(n_qubits)
        theta = np.zeros(theta_shape)
        drawing = qml.draw(circuit)(angles, theta)
        with open(out_dir / f"qnpg_vqc_{bus}_bus_circuit.txt", "w") as f:
            f.write(drawing)
            f.write("\n")


def write_markdown(out_dir: Path, qnpg: list[dict], qgnn: list[dict]):
    with open(out_dir / "quantum_architecture_summary.md", "w") as f:
        f.write("# Quantum Architecture Summary\n\n")
        f.write("## Q-NPG-FDIA VQC Actor\n\n")
        for item in qnpg:
            f.write(f"### IEEE {item['bus']}-Bus\n\n")
            f.write(f"- Qubits: {item['n_qubits']}\n")
            f.write(f"- Layers: {item['vqc_layers']}\n")
            f.write(f"- Ansatz parameters: {item['theta_parameters']}\n")
            f.write(f"- Encoding: {item['encoding']}\n")
            f.write(f"- Readout: {item['readout']}\n")
            f.write(f"- IBM verification: {item['ibm_verification']}\n\n")
        f.write("## QGNN / Quantum Detector Templates\n\n")
        for item in qgnn:
            f.write(f"### {item['name']}\n\n")
            for key in ["purpose", "encoding", "graph_operator", "readout", "first_target", "verification"]:
                f.write(f"- {key.replace('_', ' ').title()}: {item[key]}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/quantum_architectures")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qnpg = [qnpg_summary(bus) for bus in [30, 57, 118]]
    qgnn = qgnn_detector_templates()
    payload = {"qnpg_vqc_actor": qnpg, "qgnn_detector_templates": qgnn}
    with open(out_dir / "quantum_architecture_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_qnpg_circuit_drawings(out_dir, qnpg)
    write_markdown(out_dir, qnpg, qgnn)
    print(f"wrote quantum architecture summaries to {out_dir}")


if __name__ == "__main__":
    main()
