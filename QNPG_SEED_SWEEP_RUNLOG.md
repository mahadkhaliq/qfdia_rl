# Q-NPG Seed Sweep Run Log

This file tracks the paper-only Q-NPG-FDIA attacker reruns used to address the
review concern about single-seed evaluation and missing error bars.

These runs are separate from the detector/CNN/GNN/QGNN work. They use the
attacker pipeline in `main.py`, `training/qnpg_trainer.py`,
`models/vqc_policy.py`, and `environments/grid_env.py`.

## Purpose

- Run Q-NPG-FDIA across multiple random seeds.
- Increase evaluation from the original `n = 64` to `n = 256` per seed.
- Produce mean/std tables for paper error bars.

## Local Code Changes

- `main.py`
  - Added CLI overrides for:
    - `--eval-episodes`
    - `--steps-per-update`
    - `--qfim-batch`
    - `--grad-batch`
- `scripts/run_qnpg_seed_sweep_hellbender.sbatch`
  - Slurm launcher for attacker seed sweeps.
- `scripts/summarize_qnpg_seed_sweep.py`
  - Aggregates per-seed attacker CSVs into raw and mean/std summary tables.

## Output Paths

On Hellbender:

```bash
/home/mkfqm/qfdia_rl_ondemand/runs/qnpg_seed_sweep/
/home/mkfqm/qfdia_rl_ondemand/paper_tables/qnpg_seed_sweep_raw.csv
/home/mkfqm/qfdia_rl_ondemand/paper_tables/qnpg_seed_sweep_summary.csv
```

Expected local paths after pulling results:

```bash
runs/qnpg_seed_sweep/
paper_tables/qnpg_seed_sweep_raw.csv
paper_tables/qnpg_seed_sweep_summary.csv
```

## Commands

Initial intended Q-NPG seed sweep:

```bash
cd /home/mkfqm/qfdia_rl_ondemand

BUS=30 SEEDS="0 1 2" EVAL_EPISODES=256 DEVICE=lightning.qubit CONDA_ENV=synthgrad \
  sbatch scripts/run_qnpg_seed_sweep_hellbender.sbatch

BUS=57 SEEDS="0 1 2" EVAL_EPISODES=256 DEVICE=lightning.qubit CONDA_ENV=synthgrad \
  sbatch scripts/run_qnpg_seed_sweep_hellbender.sbatch

BUS=118 SEEDS="0 1 2" EVAL_EPISODES=256 DEVICE=lightning.qubit CONDA_ENV=synthgrad \
  sbatch scripts/run_qnpg_seed_sweep_hellbender.sbatch
```

The first submission failed before running because `slurm_logs/` did not exist
before Slurm opened the output/error paths. The directory was created and the
jobs were resubmitted.

## Job IDs

### Failed Startup Attempt

| Job ID | Bus | Device | State | Notes |
| --- | --- | --- | --- | --- |
| 14704227 | 30 | lightning.qubit | FAILED | `slurm_logs/` missing before Slurm output open |
| 14704228 | 57 | lightning.qubit | FAILED | `slurm_logs/` missing before Slurm output open |
| 14704229 | 118 | lightning.qubit | FAILED | `slurm_logs/` missing before Slurm output open |

### First Real Submission

| Job ID | Bus | Device | State | Notes |
| --- | --- | --- | --- | --- |
| 14704248 | 30 | lightning.qubit | FAILED | PennyLane metric tensor rejected `adjoint` on `lightning.qubit` |
| 14704249 | 57 | lightning.qubit | FAILED | PennyLane metric tensor rejected `adjoint` on `lightning.qubit` |
| 14704250 | 118 | lightning.qubit | FAILED | PennyLane metric tensor rejected `adjoint` on `lightning.qubit` |

Failure signature for 30/118:

```text
pennylane.exceptions.QuantumFunctionError:
Device <lightning.qubit ...> does not support adjoint with requested circuit.
```

### Rerun With `default.qubit`

Rerun command:

```bash
cd /home/mkfqm/qfdia_rl_ondemand

BUS=30 SEEDS="0 1 2" EVAL_EPISODES=256 DEVICE=default.qubit CONDA_ENV=synthgrad \
  sbatch scripts/run_qnpg_seed_sweep_hellbender.sbatch

BUS=118 SEEDS="0 1 2" EVAL_EPISODES=256 DEVICE=default.qubit CONDA_ENV=synthgrad \
  sbatch scripts/run_qnpg_seed_sweep_hellbender.sbatch

BUS=57 SEEDS="0 1 2" EVAL_EPISODES=256 DEVICE=default.qubit CONDA_ENV=synthgrad \
  sbatch scripts/run_qnpg_seed_sweep_hellbender.sbatch
```

| Job ID | Bus | Device | State at last check | Notes |
| --- | --- | --- | --- | --- |
| 14704515 | 30 | default.qubit | COMPLETED | Replacement for failed 30-bus lightning job; seeds 0, 1, 2 complete |
| 14704516 | 118 | default.qubit | COMPLETED | Replacement for failed 118-bus lightning job; seeds 0, 1, 2 complete |
| 14704711 | 57 | default.qubit | COMPLETED | Replacement for failed 57-bus lightning job; seeds 0, 1, 2 complete |

Final Slurm accounting:

```text
14704515  qfdia_qnpg_seed  COMPLETED  0:0  02:17:33  c039
14704516  qfdia_qnpg_seed  COMPLETED  0:0  07:11:28  c037
14704711  qfdia_qnpg_seed  COMPLETED  0:0  04:54:45  c009
```

Final pulled local artifacts:

```text
runs/qnpg_seed_sweep/bus30_seed{0,1,2}/qnpg_30_results.csv
runs/qnpg_seed_sweep/bus57_seed{0,1,2}/qnpg_57_results.csv
runs/qnpg_seed_sweep/bus118_seed{0,1,2}/qnpg_118_results.csv
paper_tables/qnpg_seed_sweep_raw.csv       # 54 raw rows + header
paper_tables/qnpg_seed_sweep_summary.csv   # 18 summary rows + header
runs/logs/qnpg_seed_14704515.{out,err}
runs/logs/qnpg_seed_14704516.{out,err}
runs/logs/qnpg_seed_14704711.{out,err}
```

## Status Check Commands

```bash
squeue -j 14704515,14704516,14704711

sacct -j 14704515,14704516,14704711 \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

Useful logs:

```bash
tail -n 120 slurm_logs/qnpg_seed_14704515.out
tail -n 120 slurm_logs/qnpg_seed_14704515.err
tail -n 120 slurm_logs/qnpg_seed_14704516.out
tail -n 120 slurm_logs/qnpg_seed_14704516.err
tail -n 120 slurm_logs/qnpg_seed_14704711.out
tail -n 120 slurm_logs/qnpg_seed_14704711.err
```

## Paper Interpretation

These completed runs can support a statement like:

> We reran Q-NPG-FDIA for three random seeds per grid and evaluated each trained
> policy over 256 deterministic attack rollouts. We report mean and standard
> deviation across seeds.

Do not mix these rows with detector/CNN/GNN/QGNN tables. These are attacker
paper results only.
