#!/bin/bash
#SBATCH --job-name=qfdia_approx_h_ablate
#SBATCH --output=slurm_logs/approx_h_ablate_%j.out
#SBATCH --error=slurm_logs/approx_h_ablate_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs paper_tables runs/approx_h_ablation

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
REL="${REL:-0.02}"
UPDATES="${UPDATES:-80}"
STEPS_PER_UPDATE="${STEPS_PER_UPDATE:-256}"
QFIM_BATCH="${QFIM_BATCH:-6}"
GRAD_BATCH="${GRAD_BATCH:-96}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
K_PERTURB="${K_PERTURB:-10}"
SEEDS="${SEEDS:-0 1 2}"
MU_VALUES="${MU_VALUES:-0 1}"
SCALES="${SCALES:-0.5 0.75 1.0}"

python scripts/approx_h_ablation.py \
  --bus "${BUS}" \
  --ablate \
  --rel "${REL}" \
  --updates "${UPDATES}" \
  --steps-per-update "${STEPS_PER_UPDATE}" \
  --qfim-batch "${QFIM_BATCH}" \
  --grad-batch "${GRAD_BATCH}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --k-perturb "${K_PERTURB}" \
  --seeds ${SEEDS} \
  --mu-values ${MU_VALUES} \
  --scales ${SCALES} \
  --csv-out "paper_tables/approx_h_ablation_${BUS}_bus.csv" \
  --scaled-csv-out "paper_tables/approx_h_ablation_${BUS}_bus_scaled.csv" \
  --out-dir runs/approx_h_ablation \
  --log-every 5
