#!/bin/bash
#SBATCH --job-name=qnpg_fdia
#SBATCH --partition=general          # adjust to your Hellbender allocation (e.g. general / requeue)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qnpg_%j.out
#SBATCH --error=logs/qnpg_%j.err
# ---- optional GPU (only helps the 118-bus lightning.gpu backend) -------------
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# -----------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs outputs

echo "Host: $(hostname)   Date: $(date)"
module purge
module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null || true

ENV_NAME=qfdia
# create a dedicated env once (PennyLane stack; separate from your qiskit 'synthgrad' env)
if ! conda env list | grep -q "/${ENV_NAME}\$"; then
    echo "Creating conda env '${ENV_NAME}'..."
    conda create -y -n ${ENV_NAME} python=3.11
    source activate ${ENV_NAME}
    pip install --no-cache-dir -r requirements.txt
    pip install --no-cache-dir numba          # big pandapower speedup on HPC
else
    source activate ${ENV_NAME}
fi

python -c "import pennylane, pandapower, scipy; print('env OK -', pennylane.__version__)"

# ---- run all three IEEE systems --------------------------------------------
# 30-bus and 57-bus run comfortably on CPU; 118-bus is heavier.
srun python main.py --bus 30  --updates 200 --seed 42 --out outputs
srun python main.py --bus 57  --updates 200 --seed 42 --out outputs
srun python main.py --bus 118 --updates 150 --seed 42 --out outputs --device lightning.qubit

echo "All runs complete. Results in outputs/"
