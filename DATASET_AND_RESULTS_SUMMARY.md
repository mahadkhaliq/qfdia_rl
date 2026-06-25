# Dataset And Detector Results Summary

## Dataset Table

All detector datasets use the shared binary schema with `label`, measurement vector `z`, optional attack vector `a`, and metadata columns.

| Dataset | System | Samples | Normal | Attack | Feature Vector | Node Features | Topology Source |
|---|---:|---:|---:|---:|---:|---:|---|
| QGrid-Synth | IEEE 30-bus | 167,000 | 83,500 | 83,500 | 120 | 30 x 4 `[P,Q,Vm,theta]` | Pandapower/MATPOWER Ybus |
| QGrid-Synth | IEEE 57-bus | 167,000 | 83,500 | 83,500 | 228 | 57 x 4 `[P,Q,Vm,theta]` | Pandapower/MATPOWER Ybus |
| QGrid-Synth | IEEE 118-bus | 167,000 | 83,500 | 83,500 | 472 | 118 x 4 `[P,Q,Vm,theta]` | Pandapower/MATPOWER Ybus |
| Ruan CAISO / STGDL | IEEE 30-bus | 60,000 | 30,000 | 30,000 | 60 | 30 x 2 `[Vm,theta]` | Bundled `AdmittanceMatrix_30.mat` |
| Ruan CAISO / STGDL | IEEE 118-bus | 60,000 | 30,000 | 30,000 | 236 | 118 x 2 `[Vm,theta]` | Bundled `AdmittanceMatrix_118.mat` |

## Detector Models Completed

| Model | Type | Notes |
|---|---|---|
| MLP | Classical vector baseline | Dense detector over standardized measurement vector. |
| 1D-CNN | Published-style deep baseline | Convolutions over ordered measurement features. |
| GCN | Graph baseline | Pure PyTorch topology-aware graph convolution. |
| W-GCN | Physics-weighted graph baseline | GCN using normalized admittance magnitude as weighted adjacency. |
| GAT | Graph attention baseline | Masked multi-head attention over physical grid neighbors. |
| QGNN | Reduced quantum graph pilot | 4/6-qubit PennyLane quantum layer over selected nodes, with weighted reduced topology and classical binary head. |
| `+\|a\|` ablation | Oracle/residual-aware synthetic upper bound | QGrid-only CNN/MLP runs using `z + \|a\|`; not a deployable feature set unless an equivalent residual estimate is available. |

Each run logs:

```text
config.json
history.csv
metrics.json
model.pt
architecture.json
architecture.txt
Slurm/session log
```

## Current Test F1 Summary

| Dataset/System | 1D-CNN | MLP | GCN | W-GCN | GAT | QGNN-pilot | QGNN-cal | QGNN-enh4 | QGNN-enh6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QGrid-Synth 30-bus | 0.9772 | 0.9699 | 0.8711 | 0.9493 | 0.8904 | 0.7262 | 0.7550 | - | - |
| QGrid-Synth 57-bus | 0.8923 | 0.8506 | 0.5325 | 0.7356 | 0.7636 | - | - | - | - |
| QGrid-Synth 118-bus | 0.8229 | 0.8158 | 0.6380 | 0.5247 | 0.5923 | - | - | - | - |
| Ruan CAISO 30-bus | 0.9963 | 0.9961 | 0.8687 | 0.9542 | 0.9033 | 0.6777 | 0.6820 | 0.6768 | 0.7530 |
| Ruan CAISO 118-bus | 0.9890 | 0.9842 | 0.7294 | 0.8474 | 0.8256 | - | - | - | - |

QGNN pilot settings:

```text
samples: 2,000 per dataset, 400 test samples
qubits: 4
quantum layers: 2
selected nodes: [3, 5, 9, 11]
reduced edges: [(0, 1), (0, 3), (1, 2)]
backend: PennyLane default.qubit simulator inside the Hellbender synthgrad environment
```

QGNN pilot, calibrated, and enhanced metrics:

| Dataset/System | Accuracy | Balanced Acc. | Precision | Recall | F1 | AUROC | AUPRC | Latency ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QGrid-Synth 30-bus, QGNN-pilot | 0.7625 | 0.7647 | 0.8750 | 0.6207 | 0.7262 | 0.8031 | 0.8604 | 7.4843 |
| QGrid-Synth 30-bus, QGNN-cal | 0.7550 | 0.7552 | 0.7665 | 0.7438 | 0.7550 | 0.8031 | 0.8604 | 7.4620 |
| Ruan CAISO 30-bus, QGNN-pilot | 0.5125 | 0.4952 | 0.5151 | 0.9903 | 0.6777 | 0.5338 | 0.6154 | 7.3833 |
| Ruan CAISO 30-bus, QGNN-cal | 0.5175 | 0.5000 | 0.5175 | 1.0000 | 0.6820 | 0.5214 | 0.6055 | 7.3027 |
| Ruan CAISO 30-bus, QGNN-enh4 | 0.5200 | 0.5036 | 0.5194 | 0.9710 | 0.6768 | 0.6689 | 0.7508 | 6.6204 |
| Ruan CAISO 30-bus, QGNN-enh6 | 0.7950 | 0.8019 | 1.0000 | 0.6039 | 0.7530 | 0.7039 | 0.8211 | 10.9661 |

QGrid residual-aware/oracle ablation:

| Dataset/System | 1D-CNN+\|a\| F1 | MLP+\|a\| F1 | Note |
|---|---:|---:|---|
| QGrid-Synth 30-bus | 0.9999 | 0.9998 | Synthetic attack magnitude included as an upper-bound feature. |
| QGrid-Synth 57-bus | 0.9998 | 0.9995 | Synthetic attack magnitude included as an upper-bound feature. |
| QGrid-Synth 118-bus | 1.0000 | 0.9872 | Synthetic attack magnitude included as an upper-bound feature. |

## Interpretation

The 1D-CNN is currently the strongest deployable detector. MLP is competitive on easier cases and is faster. W-GCN shows that admittance weighting can greatly improve the graph baseline on QGrid-Synth 30-bus and both Ruan systems, but larger QGrid systems still lag CNN/MLP. GAT improves over plain GCN on several systems but is not consistently stronger than W-GCN.

The QGrid `+|a|` ablation is an oracle/residual-aware upper bound: performance is nearly perfect because the synthetic attack magnitude is supplied. This is useful for paper analysis, but should be clearly separated from deployable detectors unless a realistic residual estimator supplies an equivalent signal.

The reduced QGNN pilot is useful as a quantum architecture proof, not yet as a winning detector. On QGrid-Synth 30-bus it learns a meaningful decision boundary, and validation threshold calibration improves F1 from 0.7262 to 0.7550. On Ruan CAISO 30-bus, the original 4-qubit calibrated model over-flags, while the enhanced 6-qubit hybrid node-selection model improves F1 to 0.7530 and reduces FPR to 0.0. A post-hoc threshold sweep on the enhanced 6-qubit model improves recall from 0.6039 to 0.6135 at FPR <= 0.05, so the remaining limitation is model separability rather than threshold choice alone.

## Plot Artifacts

Paper-writing summary:

```text
PAPER_RESULTS.md
```

Combined plots:

```text
runs/plots/detector_metrics_bars.png
runs/plots/detector_learning_curves.png
```

Individual metric plots:

```text
runs/plots/metrics/*.png
```

Individual learning curves:

```text
runs/plots/learning_curves/*.png
```

Focused per-system plots:

```text
runs/plots/by_system/*_metrics.png
runs/plots/by_system/*_learning.png
```

Focused per-bus plots:

```text
runs/plots/by_bus/*_metrics.png
runs/plots/by_bus/*_learning.png
```
