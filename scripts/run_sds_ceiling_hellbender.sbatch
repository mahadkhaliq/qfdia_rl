#!/bin/bash
#SBATCH --job-name=qfdia_sds_ceiling
#SBATCH --output=slurm_logs/sds_ceiling_%j.out
#SBATCH --error=slurm_logs/sds_ceiling_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs paper_tables

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate qfdia

python scripts/max_sds_ceiling.py --all --restarts 40 --iters 25 --csv-out paper_tables/sds_ceiling_ratios.csv
