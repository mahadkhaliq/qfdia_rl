#!/bin/bash
#SBATCH --job-name=ruan_norm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ruan_norm_%j.out
#SBATCH --error=logs/ruan_norm_%j.err

set -euo pipefail

BUS="${BUS:-30}"
ENV_NAME="${ENV_NAME:-synthgrad}"

mkdir -p logs data/canonical

module purge
module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python scripts/normalize_ruan_dataset.py \
  --root external/ruan_fdia \
  --bus "${BUS}" \
  --out "data/canonical/ruan_${BUS}.parquet"
