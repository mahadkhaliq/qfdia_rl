# Quantum Architecture Registry

This file records the quantum architectures used or proposed in the FDIA comparison workflow. Regenerate it with:

```bash
python scripts/quantum_architecture_summary.py
```

## Q-NPG-FDIA VQC Actor

| System | Qubits | VQC Layers | Ansatz | Encoding | Readout | Quantum Params |
|---|---:|---:|---|---|---|---:|
| IEEE 30-bus | 4 | 3 | PennyLane StronglyEntanglingLayers | AngleEmbedding with Y rotations after classical tanh encoder | PauliZ expectation on every qubit | 36 |
| IEEE 57-bus | 6 | 4 | PennyLane StronglyEntanglingLayers | AngleEmbedding with Y rotations after classical tanh encoder | PauliZ expectation on every qubit | 72 |
| IEEE 118-bus | 8 | 4 | PennyLane StronglyEntanglingLayers | AngleEmbedding with Y rotations after classical tanh encoder | PauliZ expectation on every qubit | 96 |

## Completed QGNN Detector Architectures

| Run | Dataset/System | Variant | Node Selection | Feature Mode | Qubits | Layers | Node Features | Quantum Params | Total Params | Threshold | F1 | AUROC | AUPRC |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qgrid30_qgnn_calibrated` | qgrid IEEE 30-bus | calibrated | topology | diffused | 4 | 2 | 4 | 24 | 189 | 0.3745 | 0.7550 | 0.8031 | 0.8604 |
| `qgrid30_qgnn_pilot` | qgrid IEEE 30-bus | pilot | topology | diffused | 4 | 2 | 4 | 24 | 189 | - | 0.7262 | 0.8031 | 0.8604 |
| `ruan30_qgnn_calibrated` | ruan IEEE 30-bus | calibrated | topology | diffused | 4 | 2 | 2 | 24 | 157 | 0.4776 | 0.6820 | 0.5214 | 0.6055 |
| `ruan30_qgnn_enhanced4` | ruan IEEE 30-bus | enhanced4 | label_shift | raw_plus_diffused | 4 | 2 | 4 | 24 | 189 | 0.4231 | 0.6768 | 0.6689 | 0.7508 |
| `ruan30_qgnn_enhanced6` | ruan IEEE 30-bus | enhanced6 | hybrid | raw_plus_diffused | 6 | 2 | 4 | 36 | 315 | 0.4446 | 0.7530 | 0.7039 | 0.8211 |
| `ruan30_qgnn_enhanced6_balacc` | ruan IEEE 30-bus | enhanced6_balacc | hybrid | raw_plus_diffused | 6 | 2 | 4 | 36 | 315 | 0.4446 | 0.7530 | 0.7039 | 0.8211 |
| `ruan30_qgnn_pilot` | ruan IEEE 30-bus | pilot | topology | diffused | 4 | 2 | 2 | 24 | 157 | - | 0.6777 | 0.5338 | 0.6154 |

## QGNN Circuit Pattern

- RY(pi * encoded_feature_i) on each selected-node qubit
- CNOT entanglers over reduced physical topology
- Rot(phi, theta, omega) trainable layer on each qubit
- PauliZ expectation readout on every qubit

## Notes

- `QGNN-cal` uses a validation-selected threshold; `QGNN-pilot` uses the original thresholding path.
- Current QGNN runs are 4/6-qubit reduced detectors, not full 30/57/118-bus quantum models.
- Real IBM execution should start with inference/verification snapshots, not full training loops.
