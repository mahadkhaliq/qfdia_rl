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


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def qgnn_run_variant(run_name: str) -> str:
    if "enhanced6_balacc" in run_name:
        return "enhanced6_balacc"
    if "enhanced6" in run_name:
        return "enhanced6"
    if "enhanced4" in run_name:
        return "enhanced4"
    if "calibrated" in run_name:
        return "calibrated"
    if "pilot" in run_name:
        return "pilot"
    return "run"


def collect_qgnn_runs(detectors_root: Path) -> list[dict]:
    runs = []
    for arch_path in sorted(detectors_root.glob("*qgnn*/architecture.json")):
        run_dir = arch_path.parent
        arch = load_json(arch_path)
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        config = load_json(config_path) if config_path.exists() else {}
        q_layers = int(arch.get("q_layers", config.get("q_layers", 0)))
        n_qubits = int(arch.get("n_qubits", config.get("n_qubits", 0)))
        quantum_weight_shape = arch.get("quantum_weight_shape", [q_layers, n_qubits, 3])
        quantum_parameters = arch.get("quantum_parameters", int(np.prod(quantum_weight_shape)))
        runs.append(
            {
                "run": run_dir.name,
                "variant": qgnn_run_variant(run_dir.name),
                "dataset": arch.get("dataset", config.get("dataset")),
                "bus": arch.get("bus", config.get("bus")),
                "n_qubits": n_qubits,
                "q_layers": q_layers,
                "q_device": arch.get("q_device", config.get("q_device")),
                "diff_method": arch.get("diff_method", config.get("diff_method")),
                "selected_nodes": arch.get("selected_nodes", config.get("selected_nodes", [])),
                "reduced_edges": arch.get("reduced_edges", config.get("reduced_edges", [])),
                "node_feature_dim": arch.get("node_feature_dim", config.get("node_feature_dim")),
                "adjacency_mode": arch.get("adjacency_mode", config.get("adjacency_mode")),
                "node_selection": arch.get("node_selection", config.get("node_selection", "topology")),
                "feature_mode": arch.get("feature_mode", config.get("feature_mode", "diffused")),
                "encoding": arch.get("encoding"),
                "circuit_sequence": arch.get(
                    "circuit_sequence",
                    [
                        "RY(pi * encoded_feature_i) on each selected-node qubit",
                        "CNOT entanglers over reduced physical topology",
                        "Rot(phi, theta, omega) trainable layer on each qubit",
                        "PauliZ expectation readout on every qubit",
                    ],
                ),
                "entangler": arch.get("entangler"),
                "readout": arch.get("readout"),
                "quantum_weight_shape": quantum_weight_shape,
                "quantum_parameters": int(quantum_parameters),
                "total_parameters": arch.get("total_parameters", config.get("total_parameters")),
                "trainable_parameters": arch.get("trainable_parameters", config.get("trainable_parameters")),
                "threshold": metrics.get("threshold", config.get("calibrated_threshold")),
                "f1": metrics.get("f1"),
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "fpr": metrics.get("fpr"),
                "fnr": metrics.get("fnr"),
                "latency_ms_per_sample": metrics.get("latency_ms_per_sample"),
                "metrics_path": str(metrics_path),
                "architecture_path": str(arch_path),
            }
        )
    return runs


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


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(out_dir: Path, qnpg: list[dict], qgnn: list[dict], qgnn_runs: list[dict]):
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
        if qgnn_runs:
            f.write("## Completed QGNN Detector Runs\n\n")
            for item in qgnn_runs:
                f.write(f"### {item['run']}\n\n")
                f.write(f"- Dataset/System: {item['dataset']} IEEE {item['bus']}-bus\n")
                f.write(f"- Variant: {item['variant']}\n")
                f.write(f"- Node selection: {item['node_selection']}\n")
                f.write(f"- Feature mode: {item['feature_mode']}\n")
                f.write(f"- Qubits/Layers: {item['n_qubits']} qubits, {item['q_layers']} quantum layers\n")
                f.write(f"- Quantum parameters: {item['quantum_parameters']} with shape {item['quantum_weight_shape']}\n")
                f.write(f"- Total trainable parameters: {item['trainable_parameters']}\n")
                f.write(f"- Selected nodes: {item['selected_nodes']}\n")
                f.write(f"- Reduced edges: {item['reduced_edges']}\n")
                f.write(f"- Encoding: {item['encoding']}\n")
                f.write(f"- Entangler: {item['entangler']}\n")
                f.write(f"- Readout: {item['readout']}\n")
                f.write(f"- Metrics: F1={fmt(item['f1'])}, AUROC={fmt(item['auroc'])}, AUPRC={fmt(item['auprc'])}, FPR={fmt(item['fpr'])}, FNR={fmt(item['fnr'])}\n\n")


def write_registry(path: Path, qnpg: list[dict], qgnn_runs: list[dict]):
    with open(path, "w") as f:
        f.write("# Quantum Architecture Registry\n\n")
        f.write("This file records the quantum architectures used or proposed in the FDIA comparison workflow. Regenerate it with:\n\n")
        f.write("```bash\npython scripts/quantum_architecture_summary.py\n```\n\n")
        f.write("## Q-NPG-FDIA VQC Actor\n\n")
        f.write("| System | Qubits | VQC Layers | Ansatz | Encoding | Readout | Quantum Params |\n")
        f.write("|---|---:|---:|---|---|---|---:|\n")
        for item in qnpg:
            f.write(
                f"| IEEE {item['bus']}-bus | {item['n_qubits']} | {item['vqc_layers']} | "
                f"{item['ansatz']} | {item['encoding']} | {item['readout']} | {item['theta_parameters']} |\n"
            )
        f.write("\n## Completed QGNN Detector Architectures\n\n")
        if not qgnn_runs:
            f.write("No completed QGNN detector runs were found under `runs/detectors`.\n")
            return
        f.write("| Run | Dataset/System | Variant | Node Selection | Feature Mode | Qubits | Layers | Node Features | Quantum Params | Total Params | Threshold | F1 | AUROC | AUPRC |\n")
        f.write("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for item in qgnn_runs:
            f.write(
                f"| `{item['run']}` | {item['dataset']} IEEE {item['bus']}-bus | {item['variant']} | "
                f"{item['node_selection']} | {item['feature_mode']} | "
                f"{item['n_qubits']} | {item['q_layers']} | {item['node_feature_dim']} | "
                f"{item['quantum_parameters']} | {item['trainable_parameters']} | {fmt(item['threshold'])} | "
                f"{fmt(item['f1'])} | {fmt(item['auroc'])} | {fmt(item['auprc'])} |\n"
            )
        f.write("\n## QGNN Circuit Pattern\n\n")
        first = qgnn_runs[0]
        for step in first["circuit_sequence"]:
            f.write(f"- {step}\n")
        f.write("\n## Notes\n\n")
        f.write("- `QGNN-cal` uses a validation-selected threshold; `QGNN-pilot` uses the original thresholding path.\n")
        f.write("- Current QGNN runs are 4-qubit reduced detectors, not full 30/57/118-bus quantum models.\n")
        f.write("- Real IBM execution should start with inference/verification snapshots, not full training loops.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/quantum_architectures")
    ap.add_argument("--detectors-root", default="runs/detectors")
    ap.add_argument("--registry-out", default="QUANTUM_ARCHITECTURE_REGISTRY.md")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qnpg = [qnpg_summary(bus) for bus in [30, 57, 118]]
    qgnn = qgnn_detector_templates()
    qgnn_runs = collect_qgnn_runs(Path(args.detectors_root))
    payload = {"qnpg_vqc_actor": qnpg, "qgnn_detector_templates": qgnn, "completed_qgnn_runs": qgnn_runs}
    with open(out_dir / "quantum_architecture_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_qnpg_circuit_drawings(out_dir, qnpg)
    write_markdown(out_dir, qnpg, qgnn, qgnn_runs)
    write_registry(Path(args.registry_out), qnpg, qgnn_runs)
    print(f"wrote quantum architecture summaries to {out_dir}")
    print(f"wrote quantum architecture registry to {args.registry_out}")


if __name__ == "__main__":
    main()
