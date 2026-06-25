#!/usr/bin/env bash
#SBATCH --job-name=qfdia_ibm_verify
#SBATCH --partition=interactive
#SBATCH --account=engineering
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/logs/ibm_verify_%j.out
#SBATCH --error=runs/logs/ibm_verify_%j.err

set -euo pipefail

BUS="${BUS:-30}"
DEVICE="${DEVICE:-aer}"
SHOTS="${SHOTS:-4096}"
N_POINTS="${N_POINTS:-16}"
ENV_NAME="${ENV_NAME:-qfdia_ibm_latest}"
POLICY="${POLICY:-runs/policies/qnpg_${BUS}_policy.npz}"
OUT_DIR="${OUT_DIR:-runs/quantum_architectures}"
IBM_BACKEND="${IBM_BACKEND:-}"
FAKE_BACKEND="${FAKE_BACKEND:-FakeKolkataV2}"
RESULT_TAG="${RESULT_TAG:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/verify_${DEVICE}_${BUS}_${RESULT_TAG}.json}"
LATEST_FILE="${OUT_DIR}/verify_${DEVICE}_${BUS}.json"

mkdir -p runs/logs "$OUT_DIR"

EXTRA_ARGS=()
if [[ "$DEVICE" == "ibm" && -n "$IBM_BACKEND" ]]; then
  EXTRA_ARGS+=(--ibm-backend "$IBM_BACKEND")
fi
if [[ "$DEVICE" == "aer_noisy" ]]; then
  EXTRA_ARGS+=(--fake-backend "$FAKE_BACKEND")
fi

echo "Running IBM verification"
echo "  BUS=$BUS"
echo "  DEVICE=$DEVICE"
echo "  SHOTS=$SHOTS"
echo "  N_POINTS=$N_POINTS"
echo "  ENV_NAME=$ENV_NAME"
echo "  POLICY=$POLICY"
echo "  RESULT_TAG=$RESULT_TAG"
echo "  OUT_FILE=$OUT_FILE"
echo "  LATEST_FILE=$LATEST_FILE"

conda run -n "$ENV_NAME" python verify_ibm.py \
  --bus "$BUS" \
  --load "$POLICY" \
  --device "$DEVICE" \
  --n-points "$N_POINTS" \
  --shots "$SHOTS" \
  --out "$OUT_FILE" \
  "${EXTRA_ARGS[@]}"

cp "$OUT_FILE" "$LATEST_FILE"
echo "  copied latest verification metrics to $LATEST_FILE"
