#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_PATH="data/Alloy-Bench.xlsx"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── 1. Prepare dataset (skip if already exists) ─────────────────────────────
if [ ! -f "${DATA_PATH}" ]; then
  echo "Converting Alloy-Bench dataset → ${DATA_PATH}"
  uv run python convert_dataset.py --output "${DATA_PATH}"
  echo ""
fi

# ── 2. Common generation parameters ─────────────────────────────────────────
COMMON_ARGS=(
  --data-path "${DATA_PATH}"
  --batch-size 50
  --temperature 0.6
  --top-p 0.95
  --top-k 20
  --max-new-tokens 2048
)

# ── 3. Run evaluations ──────────────────────────────────────────────────────
echo "=== MetalGPT-1 ==="
uv run python run_benchmark.py \
  --config configs/pure_metalgpt.json \
  "${COMMON_ARGS[@]}" \
  --output "results/results_pure_metalgpt_${TIMESTAMP}.xlsx"

echo ""
echo "=== Qwen3-32B ==="
uv run python run_benchmark.py \
  --config configs/pure_qwen3.json \
  "${COMMON_ARGS[@]}" \
  --output "results/results_pure_qwen3_${TIMESTAMP}.xlsx"

echo ""
echo "Done. Results in results/"
