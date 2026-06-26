#!/bin/bash
#SBATCH --job-name=qfdia_approx_h_diag
#SBATCH --output=slurm_logs/approx_h_diag_%j.out
#SBATCH --error=slurm_logs/approx_h_diag_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs paper_tables

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate qfdia

python scripts/approx_h_ablation.py --bus 30 --rel 0.02 --episodes 32 --k-perturb 10 --csv-out paper_tables/approx_h_diagnostic_30_bus.csv
python scripts/approx_h_ablation.py --bus 57 --rel 0.02 --episodes 32 --k-perturb 10 --csv-out paper_tables/approx_h_diagnostic_57_bus.csv
python scripts/approx_h_ablation.py --bus 118 --rel 0.02 --episodes 32 --k-perturb 10 --csv-out paper_tables/approx_h_diagnostic_118_bus.csv
