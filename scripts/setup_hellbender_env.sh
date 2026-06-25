#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-qfdia}"

module purge
module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found after loading miniconda3/anaconda modules." >&2
  echo "Run 'module avail miniconda anaconda' on Hellbender and adjust this script if needed." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install scikit-learn torch matplotlib

python - <<'PY'
import pennylane, pandapower, torch, sklearn
print("env OK")
print("pennylane", pennylane.__version__)
print("pandapower", pandapower.__version__)
print("torch", torch.__version__)
print("sklearn", sklearn.__version__)
PY
