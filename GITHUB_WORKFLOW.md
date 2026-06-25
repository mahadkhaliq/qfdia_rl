# GitHub / Hellbender Workflow

This repository should track code and lightweight documentation only.

Large generated files stay out of Git:

- `outputs/`
- `data/qgrid_synth/`
- `runs/detectors/`
- `runs/policies/`
- `external/ruan_fdia/`

## Local Push

Create a private GitHub repository in the browser, then add it as the remote:

```bash
git remote add origin git@github.com:<USER>/<REPO>.git
git push -u origin main
```

If using HTTPS:

```bash
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

## Hellbender Pull

On Hellbender:

```bash
cd ~
git clone git@github.com:<USER>/<REPO>.git qfdia_rl_git
cd qfdia_rl_git
```

If updating an existing checkout:

```bash
cd ~/qfdia_rl_git
git pull
```

Then link existing Hellbender data into the checkout:

```bash
mkdir -p data/qgrid_synth runs/policies

ln -sfn ~/qfdia_rl/outputs/qgrid_synth_30.parquet  data/qgrid_synth/qgrid_synth_30.parquet
ln -sfn ~/qfdia_rl/outputs/qgrid_synth_57.parquet  data/qgrid_synth/qgrid_synth_57.parquet
ln -sfn ~/qfdia_rl/outputs/qgrid_synth_118.parquet data/qgrid_synth/qgrid_synth_118.parquet

ln -sfn ~/qfdia_rl/outputs/qnpg_30_policy.npz  runs/policies/qnpg_30_policy.npz
ln -sfn ~/qfdia_rl/outputs/qnpg_57_policy.npz  runs/policies/qnpg_57_policy.npz
ln -sfn ~/qfdia_rl/outputs/qnpg_118_policy.npz runs/policies/qnpg_118_policy.npz
```

Submit detector jobs:

```bash
BUS=30  sbatch scripts/run_detector_cnn_hellbender.sbatch
BUS=57  sbatch scripts/run_detector_cnn_hellbender.sbatch
BUS=118 sbatch scripts/run_detector_cnn_hellbender.sbatch
```

