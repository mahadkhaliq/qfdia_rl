#!/bin/bash
#SBATCH --job-name=qfdia_qnpg_seed
#SBATCH --output=slurm_logs/qnpg_seed_%j.out
#SBATCH --error=slurm_logs/qnpg_seed_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs runs/qnpg_seed_sweep paper_tables

set +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
set -u

CONDA_ENV="${CONDA_ENV:-synthgrad}"
conda activate "${CONDA_ENV}"

BUS="${BUS:-30}"
SEEDS="${SEEDS:-0 1 2}"
UPDATES="${UPDATES:-200}"
STEPS_PER_UPDATE="${STEPS_PER_UPDATE:-1024}"
QFIM_BATCH="${QFIM_BATCH:-16}"
GRAD_BATCH="${GRAD_BATCH:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
DEVICE="${DEVICE:-default.qubit}"
CLASSICAL="${CLASSICAL:-0}"
QUICK="${QUICK:-0}"

for seed in ${SEEDS}; do
  run_dir="runs/qnpg_seed_sweep/bus${BUS}_seed${seed}"
  if [ "${CLASSICAL}" = "1" ]; then
    run_dir="${run_dir}_classical"
  fi
  mkdir -p "${run_dir}"

  args=(
    main.py
    --bus "${BUS}"
    --seed "${seed}"
    --out "${run_dir}"
    --updates "${UPDATES}"
    --steps-per-update "${STEPS_PER_UPDATE}"
    --qfim-batch "${QFIM_BATCH}"
    --grad-batch "${GRAD_BATCH}"
    --eval-episodes "${EVAL_EPISODES}"
    --device "${DEVICE}"
  )

  if [ "${CLASSICAL}" = "1" ]; then
    args+=(--classical)
  fi
  if [ "${QUICK}" = "1" ]; then
    args+=(--quick)
  fi

  python "${args[@]}"
done

python scripts/summarize_qnpg_seed_sweep.py \
  --root runs/qnpg_seed_sweep \
  --raw-out paper_tables/qnpg_seed_sweep_raw.csv \
  --summary-out paper_tables/qnpg_seed_sweep_summary.csv
