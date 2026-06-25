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

| Dataset/System | 1D-CNN | MLP | GCN | W-GCN | GAT |
|---|---:|---:|---:|---:|---:|
| QGrid-Synth 30-bus | 0.9772 | 0.9699 | 0.8711 | 0.9493 | 0.8904 |
| QGrid-Synth 57-bus | 0.8923 | 0.8506 | 0.5325 | 0.7356 | 0.7636 |
| QGrid-Synth 118-bus | 0.8229 | 0.8158 | 0.6380 | 0.5247 | 0.5923 |
| Ruan CAISO 30-bus | 0.9963 | 0.9961 | 0.8687 | 0.9542 | 0.9033 |
| Ruan CAISO 118-bus | 0.9890 | 0.9842 | 0.7294 | 0.8474 | 0.8256 |

## Interpretation

The 1D-CNN is currently the strongest overall detector. MLP is competitive on easier cases and is faster. W-GCN shows that admittance weighting can greatly improve the graph baseline on QGrid-Synth 30-bus and both Ruan systems, but larger QGrid systems still lag CNN/MLP. GAT improves over plain GCN on several systems but is not consistently stronger than W-GCN. This motivates the next graph stage: residual-aware node features and reduced QGNN-style detector experiments rather than topology-only message passing.

## Plot Artifacts

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
