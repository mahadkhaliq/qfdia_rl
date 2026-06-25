#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/qfdia_plot/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"

mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" scripts/plot_detector_results.py --root runs/detectors --out-dir runs/plots
"$PYTHON_BIN" scripts/export_paper_results.py
"$PYTHON_BIN" scripts/export_quantum_verification_results.py
"$PYTHON_BIN" scripts/verify_results_consistency.py
