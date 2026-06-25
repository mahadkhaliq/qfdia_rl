#!/usr/bin/env bash
#SBATCH --job-name=qfdia_ibm_ready
#SBATCH --partition=interactive
#SBATCH --account=engineering
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/ibm_ready_%j.out
#SBATCH --error=runs/logs/ibm_ready_%j.err

set -euo pipefail

ENV_NAME="${ENV_NAME:-qfdia_ibm_latest}"
BUS="${BUS:-30}"
MIN_QUBITS="${MIN_QUBITS:-4}"
SHOTS="${SHOTS:-1024}"
N_POINTS="${N_POINTS:-4}"
IBM_BACKEND="${IBM_BACKEND:-}"
RESULT_TAG="${RESULT_TAG:-ibm_smoke_${BUS}}"
OUT_DIR="${OUT_DIR:-runs/quantum_architectures}"
PREFLIGHT_OUT="${PREFLIGHT_OUT:-${OUT_DIR}/ibm_preflight_${ENV_NAME}.json}"
MANIFEST_OUT="${MANIFEST_OUT:-${OUT_DIR}/ibm_verification_manifest.json}"

mkdir -p runs/logs "$OUT_DIR"

echo "Running IBM readiness checks inside Slurm allocation"
echo "  ENV_NAME=$ENV_NAME"
echo "  BUS=$BUS"
echo "  MIN_QUBITS=$MIN_QUBITS"
echo "  SHOTS=$SHOTS"
echo "  N_POINTS=$N_POINTS"
echo "  IBM_BACKEND=${IBM_BACKEND:-auto}"
echo "  PREFLIGHT_OUT=$PREFLIGHT_OUT"
echo "  MANIFEST_OUT=$MANIFEST_OUT"

conda run -n "$ENV_NAME" python scripts/check_quantum_stack.py

conda run -n "$ENV_NAME" python scripts/ibm_quantum_preflight.py \
  --min-qubits "$MIN_QUBITS" \
  --out "$PREFLIGHT_OUT"

MANIFEST_ARGS=(
  --bus "$BUS"
  --preflight "$PREFLIGHT_OUT"
  --env-name "$ENV_NAME"
  --shots "$SHOTS"
  --n-points "$N_POINTS"
  --result-tag "$RESULT_TAG"
  --out "$MANIFEST_OUT"
  --md-out IBM_VERIFICATION_MANIFEST.md
)
if [[ -n "$IBM_BACKEND" ]]; then
  MANIFEST_ARGS+=(--backend "$IBM_BACKEND")
fi

conda run -n "$ENV_NAME" python scripts/ibm_verification_manifest.py "${MANIFEST_ARGS[@]}"
conda run -n "$ENV_NAME" python scripts/export_quantum_verification_results.py \
  --manifest "$MANIFEST_OUT" \
  --backend "${IBM_BACKEND:-ibm_fez}" \
  --shots "$SHOTS" \
  --n-points "$N_POINTS" \
  --result-tag "$RESULT_TAG"

echo "IBM readiness checks complete."
echo "Submit real hardware smoke test only after token rotation/reconfirmation:"
echo "  BUS=$BUS DEVICE=ibm IBM_BACKEND=<backend> ENV_NAME=$ENV_NAME SHOTS=$SHOTS N_POINTS=$N_POINTS RESULT_TAG=$RESULT_TAG sbatch scripts/run_ibm_verification_hellbender.sbatch"
