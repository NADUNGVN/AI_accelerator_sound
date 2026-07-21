#!/usr/bin/env bash
# Official UrbanSound8K paper_9_1 full-10 for KV260 DS1D pyramid mixup+EMA.
# No validation fold. Checkpoint rule: last snapshot (+ last-2 ensemble in metrics).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/kv260_ds1d_pyramid_mixup_ema_paper91.json}"
EXP_NAME="${EXP_NAME:-server3090_official_paper91_ds1d_full10_50ep}"
FOLDS="${FOLDS:-1-10}"
EPOCHS="${EPOCHS:-50}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-logs/official_paper91}"

mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/${EXP_NAME}_${ts}.log"

echo "hostname=$(hostname)"
echo "pwd=$ROOT"
echo "config=$CONFIG"
echo "exp_name=$EXP_NAME"
echo "folds=$FOLDS"
echo "epochs=$EPOCHS"
echo "log=$log"

test -f "$CONFIG" || { echo "Missing config: $CONFIG"; exit 1; }
test -d data/UrbanSound8K || {
  echo "data/UrbanSound8K missing; extract data/UrbanSound8K_on_server.tar.gz if available"
  exit 1
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

set -x
"$PYTHON_BIN" tools/run_multifold.py \
  --config "$CONFIG" \
  --exp_name "$EXP_NAME" \
  --folds "$FOLDS" \
  --epochs "$EPOCHS" \
  --analyze \
  --eval_modes \
  2>&1 | tee "$log"
set +x

echo "Done. Summary: experiments/${EXP_NAME}/multifold_summary.md"
echo "Primary metric column: Final test (= last snapshot). Secondary: Ensemble."
