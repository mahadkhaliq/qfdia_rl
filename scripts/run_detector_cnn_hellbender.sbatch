#!/bin/bash
#SBATCH --job-name=qfdia_cnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cnn_%x_%j.out
#SBATCH --error=logs/cnn_%x_%j.err

set -euo pipefail

BUS="${BUS:-118}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENV_NAME="${ENV_NAME:-qfdia}"
MODEL="${MODEL:-cnn1d}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

mkdir -p logs runs/detectors

module purge
module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "BUS=${BUS} MODEL=${MODEL} EPOCHS=${EPOCHS} MAX_SAMPLES=${MAX_SAMPLES}"

python scripts/train_detector_cnn.py \
  --data "data/qgrid_synth/qgrid_synth_${BUS}.parquet" \
  --model "${MODEL}" \
  --max-samples "${MAX_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --out "runs/detectors/qgrid${BUS}_${MODEL}_full"
