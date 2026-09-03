#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/tbl/miniforge3/envs/grasp-lgd/bin/python}"
cd "$ROOT"

export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for lambda_aff in 0.01 0.05; do
  "$PYTHON" -u research/scripts/train_lgdm_clean.py \
    --stems-tsv research/smoke-data/train_subset_10k.tsv \
    --out "outputs/lgdm_10k/lambda_aff_${lambda_aff}" \
    --epochs 15 \
    --train-ratio 0.8 \
    --batch-size 2 \
    --eval-steps 10 \
    --log-every 200 \
    --seed 42 \
    --condition-mode lsar \
    --lsar-affordance-weight "$lambda_aff" \
    --lsar-scale 0.01 \
    --lsar-fixed-scale
done
