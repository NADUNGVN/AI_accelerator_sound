#!/usr/bin/env bash
# AST teacher v3 — stage-2 from v2 best, aim best-val test > 90%.
#
# Diagnosis of v2:
#   best-val @ epoch 3: val 91.80% but test only 89.43%
#   later epochs had test 90–92% but lower val → never selected
# Strategy:
#   continue from v2 best/ with low LR full unfreeze, mild boost on machinery
#   classes (engine_idling=5, jackhammer=7), longer patient train, still select by VAL only.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/experiments/hf_cache}"
mkdir -p "$HF_HOME"

DATA_DIR="${DATA_DIR:-data/UrbanSound8K}"
EXP_NAME="${EXP_NAME:-server8000_ast_teacher_v3_stage2_sdp811_f1_20ep}"
# Default: continue from v2 best on this machine
INIT_CKPT="${INIT_CKPT:-experiments/server8000_ast_teacher_v2_sdp811_f1_30ep/fold_1/checkpoints/best}"

echo "[0] GPU + init checkpoint"
python - <<'PY'
import torch
from pathlib import Path
import os
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
init_ckpt = Path(os.environ.get("INIT_CKPT", "experiments/server8000_ast_teacher_v2_sdp811_f1_30ep/fold_1/checkpoints/best"))
if not init_ckpt.is_absolute():
    init_ckpt = Path.cwd() / init_ckpt
print("init_checkpoint exists:", init_ckpt.exists(), "->", init_ckpt)
if not init_ckpt.exists():
    raise SystemExit(
        f"Missing {init_ckpt}. Set INIT_CKPT=.../checkpoints/best or copy v2 best here."
    )
PY

echo "[1/2] Verify SDP split fingerprint..."
python tools/verify_sdp_split_fingerprint.py \
  --data_dir "$DATA_DIR" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83

echo "[2/2] Stage-2 fine-tune from v2 best (select by VAL only)..."
# class_weight_multipliers: 10 classes — boost engine_idling(5)=1.6, jackhammer(7)=1.35
python tools/finetune_ast_teacher.py \
  --data_dir "$DATA_DIR" \
  --exp_name "$EXP_NAME" \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83 \
  --init_checkpoint "$INIT_CKPT" \
  --epochs 20 \
  --batch_size 12 \
  --accum_steps 2 \
  --eval_batch_size 16 \
  --encoder_lr 3e-6 \
  --head_lr 5e-5 \
  --min_lr 1e-8 \
  --freeze_base_epochs 0 \
  --lr_warmup_epochs 1 \
  --label_smoothing 0.02 \
  --class_weight_multipliers "1.0,1.0,1.0,1.0,1.15,1.6,1.0,1.35,1.0,1.0" \
  --num_workers 10 \
  --early_stop_warmup 6 \
  --early_stop_patience 8 \
  --early_stop_min_delta 0.0003 \
  --eval_test_each_epoch \
  --hf_cache_dir "$HF_HOME" \
  --device cuda

echo "Done: experiments/${EXP_NAME}/fold_1/"
echo "Success bar: Best test (at best VAL epoch) > 90% (stretch >= 91%)."
echo "Do NOT pick max test over history as the official teacher metric."
