#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-qfdia_ibm_latest}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME already exists; leaving it in place."
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda run -n "$ENV_NAME" python -m pip install -U pip
conda run -n "$ENV_NAME" python -m pip install -U \
  pennylane \
  pennylane-lightning \
  pennylane-qiskit \
  qiskit \
  qiskit-aer \
  qiskit-ibm-runtime \
  numpy \
  scipy \
  pandas \
  pandapower \
  pyarrow

conda run -n "$ENV_NAME" python scripts/check_quantum_stack.py

echo
echo "IBM quantum environment ready: $ENV_NAME"
echo "Configure IBM credentials inside that environment/session before hardware runs."
