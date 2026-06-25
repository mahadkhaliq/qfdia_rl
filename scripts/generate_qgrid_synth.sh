#!/usr/bin/env bash
set -euo pipefail

BUS="${1:-30}"
TOTAL="${2:-500000}"
OUT_DIR="${3:-data/qgrid_synth}"
POLICY_DIR="${POLICY_DIR:-runs/policies}"
PER_OP="${PER_OP:-4}"
SEED="${SEED:-0}"

if [[ "${BUS}" != "30" && "${BUS}" != "57" && "${BUS}" != "118" ]]; then
  echo "Usage: $0 {30|57|118} [total_samples] [out_dir]" >&2
  exit 2
fi

if (( TOTAL % 2 != 0 )); then
  echo "total_samples must be even so normal/attack stay balanced." >&2
  exit 2
fi

POLICY="${POLICY_DIR}/qnpg_${BUS}_policy.npz"
if [[ ! -f "${POLICY}" ]]; then
  echo "Missing policy: ${POLICY}" >&2
  echo "Copy the Hellbender-trained qnpg_${BUS}_policy.npz into ${POLICY_DIR}/ first." >&2
  exit 1
fi

N_NORMAL=$((TOTAL / 2))
N_ATTACK=$((TOTAL / 2))
mkdir -p "${OUT_DIR}" logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/qgrid_synth_${BUS}_${TOTAL}_${STAMP}.log"

echo "Generating QGrid-Synth bus=${BUS}, total=${TOTAL} (${N_NORMAL} normal + ${N_ATTACK} attack)"
echo "Policy: ${POLICY}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG}"

python generate_dataset.py \
  --bus "${BUS}" \
  --load "${POLICY}" \
  --n-normal "${N_NORMAL}" \
  --n-attack "${N_ATTACK}" \
  --per-op "${PER_OP}" \
  --seed "${SEED}" \
  --out "${OUT_DIR}" 2>&1 | tee "${LOG}"
