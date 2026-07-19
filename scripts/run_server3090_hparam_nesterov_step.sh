#!/usr/bin/env bash
set -euo pipefail

FOLDS="${1:-1}"
EXP_NAME="${2:-server3090_pyramid_nesterov_step_fold1_200ep_es}"

python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_nesterov_step_200ep_es.json \
  --exp_name "${EXP_NAME}" \
  --folds "${FOLDS}" \
  --epochs 200 \
  --analyze \
  --eval_modes \
  --skip_existing
