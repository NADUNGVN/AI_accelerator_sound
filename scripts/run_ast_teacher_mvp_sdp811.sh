#!/usr/bin/env bash
# AST teacher v4 — verified HP from v1–v3 analysis (see docs/experiments/AST_TEACHER_HP_V4_VERIFY.md)
#
# Key insight (v2): epoch 4 had val 90.18% + test 90.69%, but best locked at
# epoch 3 (first full-FT after unfreeze) with val 91.80% / test 89.43%.
# v4: from-scratch FT + delay val-only best selection until epoch >= freeze+2.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/experiments/hf_cache}"
mkdir -p "$HF_HOME"

DATA_DIR="${DATA_DIR:-data/UrbanSound8K}"
EXP_NAME="${EXP_NAME:-server8000_ast_teacher_v4_sdp811_f1_25ep}"

FREEZE_BASE=3
# Skip first full-finetune epochs after unfreeze when updating best (VAL only).
SELECTION_START=$((FREEZE_BASE + 2))

echo "[0] GPU check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
PY

echo "[1/2] Verify SDP split fingerprint..."
python tools/verify_sdp_split_fingerprint.py \
  --data_dir "$DATA_DIR" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83

echo "[2/2] AST teacher v4 (from scratch, delayed val selection start=$SELECTION_START)..."
python tools/finetune_ast_teacher.py \
  --data_dir "$DATA_DIR" \
  --exp_name "$EXP_NAME" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83 \
  --epochs 25 \
  --batch_size 12 \
  --accum_steps 2 \
  --eval_batch_size 16 \
  --encoder_lr 5e-6 \
  --head_lr 3e-4 \
  --min_lr 1e-8 \
  --freeze_base_epochs "$FREEZE_BASE" \
  --best_selection_start_epoch "$SELECTION_START" \
  --lr_warmup_epochs 2 \
  --weight_decay 0.03 \
  --label_smoothing 0.04 \
  --num_workers 10 \
  --early_stop_warmup 12 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.0005 \
  --eval_test_each_epoch \
  --hf_cache_dir "$HF_HOME" \
  --device cuda

echo "Done: experiments/${EXP_NAME}/fold_1/"
echo "Official metric: Best test AT best VAL epoch (after selection start)."
echo "Bar: > 89.89% local; target >= 90%."
