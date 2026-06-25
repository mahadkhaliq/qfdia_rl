# QGrid-Synth Data Workflow

## Directory Layout

- `runs/policies/`: trained Q-NPG policy files copied from Hellbender/cloud.
- `data/qgrid_synth/`: generated QGrid-Synth Parquet files.
- `data/canonical/`: normalized detector-ready files.
- `external/`: public datasets and downloaded references.
- `logs/`: dataset-generation logs.

## Required Policy Files

Copy the Hellbender-trained policies here:

```text
runs/policies/qnpg_30_policy.npz
runs/policies/qnpg_57_policy.npz
runs/policies/qnpg_118_policy.npz
```

## Generate 500k Samples Per Bus

Each command generates a balanced dataset:

```text
250,000 normal + 250,000 attack = 500,000 total
```

Run:

```bash
source /opt/anaconda3/bin/activate qfdia

bash scripts/generate_qgrid_synth.sh 30 500000
bash scripts/generate_qgrid_synth.sh 57 500000
bash scripts/generate_qgrid_synth.sh 118 500000
```

For a pilot:

```bash
bash scripts/generate_qgrid_synth.sh 30 2000
```

The script refuses to run if the matching policy file is missing.

## First Detector Baseline: 1D-CNN

The first published-style deep detector baseline is a 1D-CNN over the ordered
measurement vector `z`.

Pilot:

```bash
python scripts/train_detector_cnn.py \
  --data data/qgrid_synth/qgrid_synth_30.parquet \
  --model cnn1d \
  --max-samples 20000 \
  --epochs 3 \
  --out runs/detectors/qgrid30_cnn1d_pilot
```

Full within-dataset runs:

```bash
python scripts/train_detector_cnn.py \
  --data data/qgrid_synth/qgrid_synth_30.parquet \
  --model cnn1d \
  --max-samples 0 \
  --epochs 20 \
  --out runs/detectors/qgrid30_cnn1d_full

python scripts/train_detector_cnn.py \
  --data data/qgrid_synth/qgrid_synth_57.parquet \
  --model cnn1d \
  --max-samples 0 \
  --epochs 20 \
  --out runs/detectors/qgrid57_cnn1d_full

python scripts/train_detector_cnn.py \
  --data data/qgrid_synth/qgrid_synth_118.parquet \
  --model cnn1d \
  --max-samples 0 \
  --epochs 20 \
  --out runs/detectors/qgrid118_cnn1d_full
```

Primary detector metrics are F1, AUROC, AUPRC, FPR, FNR, MCC, balanced
accuracy, and latency per sample.

## Hellbender / Open OnDemand Workflow

Use Open OnDemand when SSH setup is annoying or when you want VSCode/Jupyter
inside the cluster:

- Researcher OnDemand: `https://ondemand.rnet.missouri.edu`
- Classes OnDemand: `https://hb-classes.missouri.edu`

Open a VSCode or shell session through OnDemand, then work from a project
directory on Hellbender storage. Do not run training on a login node.

One-time environment setup:

```bash
cd /path/to/qfdia_rl
bash scripts/setup_hellbender_env.sh
```

Submit detector jobs with Slurm:

```bash
BUS=30  sbatch scripts/run_detector_cnn_hellbender.sbatch
BUS=57  sbatch scripts/run_detector_cnn_hellbender.sbatch
BUS=118 sbatch scripts/run_detector_cnn_hellbender.sbatch
```

Monitor:

```bash
squeue --me
tail -f logs/cnn_qfdia_cnn_<jobid>.out
```

Cancel if needed:

```bash
scancel <jobid>
```
