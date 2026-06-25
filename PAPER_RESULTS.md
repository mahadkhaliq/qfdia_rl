# Paper Results: FDIA Detector Comparison

Regenerate this file after new detector runs with:

```bash
python scripts/export_paper_results.py
```

## Best Deployable Detector Per Dataset/System

| Dataset/System | Best Model | F1 | AUROC | AUPRC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- |
| QGrid-Synth 118-bus | 1D-CNN | 0.8229 | 0.8967 | 0.9229 | 0.1005 | 0.2307 |
| QGrid-Synth 30-bus | 1D-CNN | 0.9772 | 0.9964 | 0.9971 | 0.0110 | 0.0341 |
| QGrid-Synth 57-bus | 1D-CNN | 0.8923 | 0.9529 | 0.9646 | 0.0543 | 0.1507 |
| Ruan CAISO 118-bus | 1D-CNN | 0.9890 | 0.9994 | 0.9995 | 0.0000 | 0.0218 |
| Ruan CAISO 30-bus | 1D-CNN | 0.9963 | 0.9979 | 0.9986 | 0.0000 | 0.0074 |

## Best Detector Per Dataset/System

| Dataset/System | Best Model | F1 | AUROC | AUPRC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- |
| QGrid-Synth 118-bus | 1D-CNN+\|a\| | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| QGrid-Synth 30-bus | 1D-CNN+\|a\| | 0.9999 | 1.0000 | 1.0000 | 0.0000 | 0.0001 |
| QGrid-Synth 57-bus | 1D-CNN+\|a\| | 0.9998 | 1.0000 | 1.0000 | 0.0000 | 0.0005 |
| Ruan CAISO 118-bus | 1D-CNN | 0.9890 | 0.9994 | 0.9995 | 0.0000 | 0.0218 |
| Ruan CAISO 30-bus | 1D-CNN | 0.9963 | 0.9979 | 0.9986 | 0.0000 | 0.0074 |

## Metrics By Bus Number

### IEEE 30-Bus

| Dataset/System | Model | F1 | Balanced Acc. | Precision | Recall | MCC | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QGrid-Synth 30-bus | 1D-CNN | 0.9772 | 0.9775 | 0.9888 | 0.9659 | 0.9552 | 0.9964 | 0.9971 |
| QGrid-Synth 30-bus | 1D-CNN+\|a\| | 0.9999 | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 1.0000 |
| QGrid-Synth 30-bus | MLP | 0.9699 | 0.9700 | 0.9719 | 0.9680 | 0.9399 | 0.9952 | 0.9960 |
| QGrid-Synth 30-bus | MLP+\|a\| | 0.9998 | 0.9998 | 1.0000 | 0.9996 | 0.9996 | 1.0000 | 1.0000 |
| QGrid-Synth 30-bus | GCN | 0.8711 | 0.8852 | 0.9933 | 0.7757 | 0.7896 | 0.9660 | 0.9739 |
| QGrid-Synth 30-bus | W-GCN | 0.9493 | 0.9512 | 0.9864 | 0.9149 | 0.9047 | 0.9865 | 0.9897 |
| QGrid-Synth 30-bus | GAT | 0.8904 | 0.9012 | 1.0000 | 0.8025 | 0.8186 | 0.9864 | 0.9898 |
| QGrid-Synth 30-bus | QGNN-pilot | 0.7262 | 0.7647 | 0.8750 | 0.6207 | 0.5513 | 0.8031 | 0.8604 |
| QGrid-Synth 30-bus | QGNN-cal | 0.7550 | 0.7552 | 0.7665 | 0.7438 | 0.5103 | 0.8031 | 0.8604 |
| Ruan CAISO 30-bus | 1D-CNN | 0.9963 | 0.9963 | 1.0000 | 0.9926 | 0.9926 | 0.9979 | 0.9986 |
| Ruan CAISO 30-bus | MLP | 0.9961 | 0.9961 | 1.0000 | 0.9922 | 0.9922 | 0.9981 | 0.9988 |
| Ruan CAISO 30-bus | GCN | 0.8687 | 0.8836 | 0.9961 | 0.7702 | 0.7877 | 0.9675 | 0.9752 |
| Ruan CAISO 30-bus | W-GCN | 0.9542 | 0.9557 | 0.9884 | 0.9222 | 0.9135 | 0.9746 | 0.9829 |
| Ruan CAISO 30-bus | GAT | 0.9033 | 0.9118 | 1.0000 | 0.8236 | 0.8367 | 0.9706 | 0.9794 |
| Ruan CAISO 30-bus | QGNN-pilot | 0.6777 | 0.4952 | 0.5151 | 0.9903 | -0.0684 | 0.5338 | 0.6154 |
| Ruan CAISO 30-bus | QGNN-cal | 0.6820 | 0.5000 | 0.5175 | 1.0000 | 0.0000 | 0.5214 | 0.6055 |
| Ruan CAISO 30-bus | QGNN-enh4 | 0.6768 | 0.5036 | 0.5194 | 0.9710 | 0.0205 | 0.6689 | 0.7508 |
| Ruan CAISO 30-bus | QGNN-enh6 | 0.7530 | 0.8019 | 1.0000 | 0.6039 | 0.6510 | 0.7039 | 0.8211 |
| Ruan CAISO 30-bus | QGNN-enh6-BA | 0.7530 | 0.8019 | 1.0000 | 0.6039 | 0.6510 | 0.7039 | 0.8211 |

### IEEE 57-Bus

| Dataset/System | Model | F1 | Balanced Acc. | Precision | Recall | MCC | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QGrid-Synth 57-bus | 1D-CNN | 0.8923 | 0.8975 | 0.9400 | 0.8493 | 0.7988 | 0.9529 | 0.9646 |
| QGrid-Synth 57-bus | 1D-CNN+\|a\| | 0.9998 | 0.9998 | 1.0000 | 0.9995 | 0.9995 | 1.0000 | 1.0000 |
| QGrid-Synth 57-bus | MLP | 0.8506 | 0.8629 | 0.9347 | 0.7804 | 0.7360 | 0.9175 | 0.9407 |
| QGrid-Synth 57-bus | MLP+\|a\| | 0.9995 | 0.9995 | 1.0000 | 0.9990 | 0.9990 | 1.0000 | 1.0000 |
| QGrid-Synth 57-bus | GCN | 0.5325 | 0.6799 | 0.9867 | 0.3646 | 0.4634 | 0.7774 | 0.8303 |
| QGrid-Synth 57-bus | W-GCN | 0.7356 | 0.7862 | 0.9640 | 0.5947 | 0.6197 | 0.8680 | 0.9008 |
| QGrid-Synth 57-bus | GAT | 0.7636 | 0.8070 | 0.9849 | 0.6235 | 0.6600 | 0.8714 | 0.9084 |

### IEEE 118-Bus

| Dataset/System | Model | F1 | Balanced Acc. | Precision | Recall | MCC | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QGrid-Synth 118-bus | 1D-CNN | 0.8229 | 0.8344 | 0.8845 | 0.7693 | 0.6745 | 0.8967 | 0.9229 |
| QGrid-Synth 118-bus | 1D-CNN+\|a\| | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| QGrid-Synth 118-bus | MLP | 0.8158 | 0.8385 | 0.9493 | 0.7153 | 0.6986 | 0.8602 | 0.9056 |
| QGrid-Synth 118-bus | MLP+\|a\| | 0.9872 | 0.9873 | 0.9997 | 0.9750 | 0.9750 | 0.9986 | 0.9989 |
| QGrid-Synth 118-bus | GCN | 0.6380 | 0.7160 | 0.8797 | 0.5005 | 0.4788 | 0.7626 | 0.8180 |
| QGrid-Synth 118-bus | W-GCN | 0.5247 | 0.6690 | 0.9299 | 0.3654 | 0.4252 | 0.7370 | 0.7900 |
| QGrid-Synth 118-bus | GAT | 0.5923 | 0.6764 | 0.8002 | 0.4702 | 0.3873 | 0.7380 | 0.7821 |
| Ruan CAISO 118-bus | 1D-CNN | 0.9890 | 0.9891 | 1.0000 | 0.9782 | 0.9784 | 0.9994 | 0.9995 |
| Ruan CAISO 118-bus | MLP | 0.9842 | 0.9844 | 1.0000 | 0.9688 | 0.9693 | 0.9971 | 0.9976 |
| Ruan CAISO 118-bus | GCN | 0.7294 | 0.7686 | 0.8786 | 0.6235 | 0.5614 | 0.8225 | 0.8747 |
| Ruan CAISO 118-bus | W-GCN | 0.8474 | 0.8606 | 0.9362 | 0.7740 | 0.7323 | 0.8927 | 0.9281 |
| Ruan CAISO 118-bus | GAT | 0.8256 | 0.8515 | 1.0000 | 0.7031 | 0.7362 | 0.9046 | 0.9363 |

## Overall Detector Metrics

| Dataset/System | Model | F1 | AUROC | AUPRC | FPR | FNR | Latency ms/sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QGrid-Synth 30-bus | 1D-CNN | 0.9772 | 0.9964 | 0.9971 | 0.0110 | 0.0341 | 0.0056 |
| QGrid-Synth 30-bus | 1D-CNN+\|a\| | 0.9999 | 1.0000 | 1.0000 | 0.0000 | 0.0001 | 0.0058 |
| QGrid-Synth 30-bus | MLP | 0.9699 | 0.9952 | 0.9960 | 0.0280 | 0.0320 | 0.0046 |
| QGrid-Synth 30-bus | MLP+\|a\| | 0.9998 | 1.0000 | 1.0000 | 0.0000 | 0.0004 | 0.0049 |
| QGrid-Synth 30-bus | GCN | 0.8711 | 0.9660 | 0.9739 | 0.0053 | 0.2243 | 0.0054 |
| QGrid-Synth 30-bus | W-GCN | 0.9493 | 0.9865 | 0.9897 | 0.0126 | 0.0851 | 0.0055 |
| QGrid-Synth 30-bus | GAT | 0.8904 | 0.9864 | 0.9898 | 0.0000 | 0.1975 | 0.0063 |
| QGrid-Synth 30-bus | QGNN-pilot | 0.7262 | 0.8031 | 0.8604 | 0.0914 | 0.3793 | 7.4843 |
| QGrid-Synth 30-bus | QGNN-cal | 0.7550 | 0.8031 | 0.8604 | 0.2335 | 0.2562 | 7.4620 |
| QGrid-Synth 57-bus | 1D-CNN | 0.8923 | 0.9529 | 0.9646 | 0.0543 | 0.1507 | 0.0059 |
| QGrid-Synth 57-bus | 1D-CNN+\|a\| | 0.9998 | 1.0000 | 1.0000 | 0.0000 | 0.0005 | 0.0077 |
| QGrid-Synth 57-bus | MLP | 0.8506 | 0.9175 | 0.9407 | 0.0546 | 0.2196 | 0.0051 |
| QGrid-Synth 57-bus | MLP+\|a\| | 0.9995 | 1.0000 | 1.0000 | 0.0000 | 0.0010 | 0.0059 |
| QGrid-Synth 57-bus | GCN | 0.5325 | 0.7774 | 0.8303 | 0.0049 | 0.6354 | 0.0056 |
| QGrid-Synth 57-bus | W-GCN | 0.7356 | 0.8680 | 0.9008 | 0.0222 | 0.4053 | 0.0057 |
| QGrid-Synth 57-bus | GAT | 0.7636 | 0.8714 | 0.9084 | 0.0096 | 0.3765 | 0.0068 |
| QGrid-Synth 118-bus | 1D-CNN | 0.8229 | 0.8967 | 0.9229 | 0.1005 | 0.2307 | 0.0067 |
| QGrid-Synth 118-bus | 1D-CNN+\|a\| | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0092 |
| QGrid-Synth 118-bus | MLP | 0.8158 | 0.8602 | 0.9056 | 0.0382 | 0.2847 | 0.0057 |
| QGrid-Synth 118-bus | MLP+\|a\| | 0.9872 | 0.9986 | 0.9989 | 0.0003 | 0.0250 | 0.0058 |
| QGrid-Synth 118-bus | GCN | 0.6380 | 0.7626 | 0.8180 | 0.0684 | 0.4995 | 0.0064 |
| QGrid-Synth 118-bus | W-GCN | 0.5247 | 0.7370 | 0.7900 | 0.0275 | 0.6346 | 0.0066 |
| QGrid-Synth 118-bus | GAT | 0.5923 | 0.7380 | 0.7821 | 0.1174 | 0.5298 | 0.0112 |
| Ruan CAISO 30-bus | 1D-CNN | 0.9963 | 0.9979 | 0.9986 | 0.0000 | 0.0074 | 0.0085 |
| Ruan CAISO 30-bus | MLP | 0.9961 | 0.9981 | 0.9988 | 0.0000 | 0.0078 | 0.0048 |
| Ruan CAISO 30-bus | GCN | 0.8687 | 0.9675 | 0.9752 | 0.0030 | 0.2298 | 0.0076 |
| Ruan CAISO 30-bus | W-GCN | 0.9542 | 0.9746 | 0.9829 | 0.0108 | 0.0778 | 0.0054 |
| Ruan CAISO 30-bus | GAT | 0.9033 | 0.9706 | 0.9794 | 0.0000 | 0.1764 | 0.0061 |
| Ruan CAISO 30-bus | QGNN-pilot | 0.6777 | 0.5338 | 0.6154 | 1.0000 | 0.0097 | 7.3833 |
| Ruan CAISO 30-bus | QGNN-cal | 0.6820 | 0.5214 | 0.6055 | 1.0000 | 0.0000 | 7.3027 |
| Ruan CAISO 30-bus | QGNN-enh4 | 0.6768 | 0.6689 | 0.7508 | 0.9637 | 0.0290 | 6.6204 |
| Ruan CAISO 30-bus | QGNN-enh6 | 0.7530 | 0.7039 | 0.8211 | 0.0000 | 0.3961 | 10.9661 |
| Ruan CAISO 30-bus | QGNN-enh6-BA | 0.7530 | 0.7039 | 0.8211 | 0.0000 | 0.3961 | 10.7415 |
| Ruan CAISO 118-bus | 1D-CNN | 0.9890 | 0.9994 | 0.9995 | 0.0000 | 0.0218 | 0.0062 |
| Ruan CAISO 118-bus | MLP | 0.9842 | 0.9971 | 0.9976 | 0.0000 | 0.0312 | 0.0051 |
| Ruan CAISO 118-bus | GCN | 0.7294 | 0.8225 | 0.8747 | 0.0862 | 0.3765 | 0.0059 |
| Ruan CAISO 118-bus | W-GCN | 0.8474 | 0.8927 | 0.9281 | 0.0528 | 0.2260 | 0.0062 |
| Ruan CAISO 118-bus | GAT | 0.8256 | 0.9046 | 0.9363 | 0.0000 | 0.2969 | 0.0111 |

## QGNN Threshold Sweep Summaries

These rows report post-hoc threshold choices on saved QGNN test probabilities for recall/FPR tradeoff analysis.

| Run | Criterion | Threshold | F1 | Balanced Acc. | Precision | Recall | FPR | FNR | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ruan30_qgnn_enhanced6` | Best F1 | 0.4446 | 0.7568 | 0.8043 | 1.0000 | 0.6087 | 0.0000 | 0.3913 | 0.6548 |
| `ruan30_qgnn_enhanced6` | Best Balanced Acc. | 0.4446 | 0.7568 | 0.8043 | 1.0000 | 0.6087 | 0.0000 | 0.3913 | 0.6548 |
| `ruan30_qgnn_enhanced6` | Best MCC | 0.4446 | 0.7568 | 0.8043 | 1.0000 | 0.6087 | 0.0000 | 0.3913 | 0.6548 |
| `ruan30_qgnn_enhanced6` | Best Recall @ FPR <= 0.05 | 0.4446 | 0.7405 | 0.7834 | 0.9338 | 0.6135 | 0.0466 | 0.3865 | 0.5980 |
| `ruan30_qgnn_enhanced6` | Best Recall @ FPR <= 0.10 | 0.4446 | 0.7405 | 0.7834 | 0.9338 | 0.6135 | 0.0466 | 0.3865 | 0.5980 |
| `ruan30_qgnn_enhanced6` | Best Recall @ FPR <= 0.20 | 0.4445 | 0.6957 | 0.7237 | 0.7950 | 0.6184 | 0.1710 | 0.3816 | 0.4558 |

## Q-NPG Analytical Stealth Ceiling

These rows compare the learned Q-NPG-FDIA mean SDS against a direct max-SDS optimizer under the same linearized stealth constraint `a = Hc` and per-meter box bound.

| Bus | a_max | SDS Ceiling | Learned SDS | Learned/Ceiling | Stealth Residual | BDD Tau |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.3000 | 2.3238 | 0.4701 | 0.2023 | 2.62e-23 | 79.082 |
| 57 | 0.1800 | 1.9043 | 0.2167 | 0.1138 | 1.12e-23 | 139.921 |
| 118 | 0.1200 | 1.7362 | 0.0722 | 0.0416 | 1.18e-22 | 272.836 |

## Complete Table Files

- `paper_tables/best_deployable_by_system.csv`
- `paper_tables/detector_metrics_118_bus.csv`
- `paper_tables/detector_metrics_30_bus.csv`
- `paper_tables/detector_metrics_57_bus.csv`
- `paper_tables/detector_metrics_complete.csv`
- `paper_tables/detector_metrics_deployable.csv`
- `paper_tables/detector_metrics_oracle_ablations.csv`
- `paper_tables/quantum_verification_results.csv`
- `paper_tables/sds_ceiling_ratios.csv`

## Figure Index

### Combined Figures

- `runs/plots/detector_metrics_bars.png`
- `runs/plots/detector_learning_curves.png`

### Paper Figures

- `runs/plots/paper/auprc_heatmap_deployable.png`
- `runs/plots/paper/auprc_heatmap_with_oracle.png`
- `runs/plots/paper/auroc_heatmap_deployable.png`
- `runs/plots/paper/auroc_heatmap_with_oracle.png`
- `runs/plots/paper/f1_heatmap_deployable.png`
- `runs/plots/paper/f1_heatmap_with_oracle.png`
- `runs/plots/paper/f1_ranked_by_system_deployable.png`
- `runs/plots/paper/fpr_fnr_tradeoff_deployable.png`
- `runs/plots/paper/latency_deployable.png`

### Per-Bus Figures

- `runs/plots/by_bus/118_bus_learning.png`
- `runs/plots/by_bus/118_bus_metrics.png`
- `runs/plots/by_bus/30_bus_learning.png`
- `runs/plots/by_bus/30_bus_metrics.png`
- `runs/plots/by_bus/57_bus_learning.png`
- `runs/plots/by_bus/57_bus_metrics.png`

### Per-Bus / Per-Method Figures

- `runs/plots/by_bus_model/118_bus_cnn1d_learning.png`
- `runs/plots/by_bus_model/118_bus_cnn1d_metrics.png`
- `runs/plots/by_bus_model/118_bus_gat_learning.png`
- `runs/plots/by_bus_model/118_bus_gat_metrics.png`
- `runs/plots/by_bus_model/118_bus_gcn_learning.png`
- `runs/plots/by_bus_model/118_bus_gcn_metrics.png`
- `runs/plots/by_bus_model/118_bus_mlp_learning.png`
- `runs/plots/by_bus_model/118_bus_mlp_metrics.png`
- `runs/plots/by_bus_model/118_bus_wgcn_learning.png`
- `runs/plots/by_bus_model/118_bus_wgcn_metrics.png`
- `runs/plots/by_bus_model/30_bus_cnn1d_learning.png`
- `runs/plots/by_bus_model/30_bus_cnn1d_metrics.png`
- `runs/plots/by_bus_model/30_bus_gat_learning.png`
- `runs/plots/by_bus_model/30_bus_gat_metrics.png`
- `runs/plots/by_bus_model/30_bus_gcn_learning.png`
- `runs/plots/by_bus_model/30_bus_gcn_metrics.png`
- `runs/plots/by_bus_model/30_bus_mlp_learning.png`
- `runs/plots/by_bus_model/30_bus_mlp_metrics.png`
- `runs/plots/by_bus_model/30_bus_qgnn_learning.png`
- `runs/plots/by_bus_model/30_bus_qgnn_metrics.png`
- `runs/plots/by_bus_model/30_bus_wgcn_learning.png`
- `runs/plots/by_bus_model/30_bus_wgcn_metrics.png`
- `runs/plots/by_bus_model/57_bus_cnn1d_learning.png`
- `runs/plots/by_bus_model/57_bus_cnn1d_metrics.png`
- `runs/plots/by_bus_model/57_bus_gat_learning.png`
- `runs/plots/by_bus_model/57_bus_gat_metrics.png`
- `runs/plots/by_bus_model/57_bus_gcn_learning.png`
- `runs/plots/by_bus_model/57_bus_gcn_metrics.png`
- `runs/plots/by_bus_model/57_bus_mlp_learning.png`
- `runs/plots/by_bus_model/57_bus_mlp_metrics.png`
- `runs/plots/by_bus_model/57_bus_wgcn_learning.png`
- `runs/plots/by_bus_model/57_bus_wgcn_metrics.png`

### Per-System Figures

- `runs/plots/by_system/qgrid_118_learning.png`
- `runs/plots/by_system/qgrid_118_metrics.png`
- `runs/plots/by_system/qgrid_30_learning.png`
- `runs/plots/by_system/qgrid_30_metrics.png`
- `runs/plots/by_system/qgrid_57_learning.png`
- `runs/plots/by_system/qgrid_57_metrics.png`
- `runs/plots/by_system/ruan_118_learning.png`
- `runs/plots/by_system/ruan_118_metrics.png`
- `runs/plots/by_system/ruan_30_learning.png`
- `runs/plots/by_system/ruan_30_metrics.png`

## Metric Definitions For Paper

- `+|a|` model labels denote QGrid-only oracle/residual-aware ablations using the synthetic attack vector magnitude.
- F1: harmonic mean of precision and recall for attack detection.
- AUROC: threshold-independent separability between normal and attack samples.
- AUPRC: precision-recall area, useful when attack/normal ratios change.
- FPR: normal samples incorrectly flagged as attacks.
- FNR: attacks missed by the detector.
- MCC: balanced correlation-style score over all confusion-matrix cells.
- Latency: measured inference time per sample in the recorded run environment.
