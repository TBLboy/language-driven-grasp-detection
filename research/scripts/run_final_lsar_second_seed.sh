#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/tbl/miniforge3/envs/grasp-lgd/bin/python}"
cd "$ROOT"

export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Run the frozen LSAR final method with a second training seed.
# Expected: LSAR_LAMBDA={0.0,0.01,0.05,0.1}, SEED=43, SPLIT_SEED=42.
LSAR_LAMBDA="${LSAR_LAMBDA:-0.0}"
SEED="${SEED:-43}"
SPLIT_SEED="${SPLIT_SEED:-42}"
OUT="outputs/lgdm_10k/lsar_final_lambda_${LSAR_LAMBDA}_seed${SEED}_split${SPLIT_SEED}"

"$PYTHON" -u research/scripts/train_lgdm_clean.py \
  --stems-tsv research/smoke-data/train_subset_10k.tsv \
  --out "$OUT" \
  --epochs 15 \
  --train-ratio 0.8 \
  --batch-size 2 \
  --eval-steps 10 \
  --log-every 200 \
  --seed "$SEED" \
  --split-seed "$SPLIT_SEED" \
  --condition-mode lsar \
  --lsar-affordance-weight "$LSAR_LAMBDA" \
  --lsar-scale 0.01 \
  --lsar-fixed-scale
