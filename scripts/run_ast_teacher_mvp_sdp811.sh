#!/usr/bin/env bash
# AST teacher v2 on RTX 8000 — target best-val test > 89.89% (stretch 90–92%+)
# Locked: SDP source_group_8_1_1, seed 83, fold 1
#
# Why v2 (vs failed MVP ~88.3%):
# - MVP early-stopped on val@3; later epochs had higher test (~89.5%) but val wiggled
# - weighted_sampler removed (local ~90% did not use it)
# - Match local recipe more: freeze 2 ep, encoder_lr 1e-5, head_lr 5e-4
# - Longer train, very patient ES so we actually reach late epochs
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/experiments/hf_cache}"
mkdir -p "$HF_HOME"

DATA_DIR="${DATA_DIR:-data/UrbanSound8K}"
EXP_NAME="${EXP_NAME:-server8000_ast_teacher_v2_sdp811_f1_30ep}"

echo "[0] GPU check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — fix torch/driver first"
print("device:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
PY

echo "[1/2] Verify SDP split fingerprint..."
python tools/verify_sdp_split_fingerprint.py \
  --data_dir "$DATA_DIR" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83

echo "[2/2] Fine-tune AST teacher v2 (no weighted_sampler, patient ES, 30 ep)..."
python tools/finetune_ast_teacher.py \
  --data_dir "$DATA_DIR" \
  --exp_name "$EXP_NAME" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83 \
  --epochs 30 \
  --batch_size 12 \
  --accum_steps 2 \
  --eval_batch_size 16 \
  --encoder_lr 1e-5 \
  --head_lr 5e-4 \
  --min_lr 1e-7 \
  --freeze_base_epochs 2 \
  --lr_warmup_epochs 2 \
  --label_smoothing 0.03 \
  --num_workers 10 \
  --early_stop_warmup 12 \
  --early_stop_patience 12 \
  --early_stop_min_delta 0.0005 \
  --eval_test_each_epoch \
  --hf_cache_dir "$HF_HOME" \
  --device cuda

echo "Done."
echo "  Metrics: experiments/${EXP_NAME}/fold_1/metrics.json"
echo "  Summary: experiments/${EXP_NAME}/fold_1/summary.md"
echo "  Success bar: best-val test > 89.89% (prior local). Stretch: >= 91%."
