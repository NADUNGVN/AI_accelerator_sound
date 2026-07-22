#!/usr/bin/env bash
# AST teacher MVP on RTX 8000 — locked SDP 8-1-1 seed 83 fold 1
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/experiments/hf_cache}"
mkdir -p "$HF_HOME"

DATA_DIR="${DATA_DIR:-data/UrbanSound8K}"
EXP_NAME="${EXP_NAME:-server8000_ast_teacher_mvp_sdp811_f1_24ep}"

echo "[1/2] Verify SDP split fingerprint..."
python tools/verify_sdp_split_fingerprint.py \
  --data_dir "$DATA_DIR" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83

echo "[2/2] Fine-tune AST teacher..."
python tools/finetune_ast_teacher.py \
  --data_dir "$DATA_DIR" \
  --exp_name "$EXP_NAME" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83 \
  --epochs 24 \
  --batch_size 8 \
  --accum_steps 2 \
  --eval_batch_size 16 \
  --encoder_lr 1.5e-5 \
  --head_lr 3e-4 \
  --freeze_base_epochs 1 \
  --lr_warmup_epochs 2 \
  --weighted_sampler \
  --num_workers 8 \
  --early_stop_warmup 8 \
  --early_stop_patience 6 \
  --eval_test_each_epoch \
  --hf_cache_dir "$HF_HOME" \
  --device cuda

echo "Done. Metrics: experiments/${EXP_NAME}/fold_1/metrics.json"
echo "Summary: experiments/${EXP_NAME}/fold_1/summary.md"
