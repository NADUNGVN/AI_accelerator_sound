#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON:-python}"
FOLDS="1-10"
EPOCHS="200"
BATCH_SIZE=""
SKIP_EXISTING="1"
LOG_ROOT="logs/server3090_full10"

usage() {
  cat <<'USAGE'
Run the three server-3090 full-10 experiments sequentially.

Usage:
  bash scripts/run_server3090_full10_three_models.sh [options]

Options:
  --python PATH        Python executable. Default: python
  --folds LIST        Folds to run, e.g. 1-10 or 1,2,3. Default: 1-10
  --epochs N          Planned epochs per fold. Default: 200
  --batch-size N      Override config batch_size, e.g. 64 if CUDA OOM occurs.
  --no-skip-existing  Re-run folds even if metrics.json already exists.
  -h, --help          Show this help.

Notes:
  - Run this after activating the Python/conda environment.
  - The protected fine-tune run depends on the baseline run checkpoints.
  - Logs are written to logs/server3090_full10/.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."
mkdir -p "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_CONFIG="configs/server3090_kv260_ds1d_pyramid_mixup_ema_200ep_es.json"
AVGMAX_CONFIG="configs/server3090_kv260_ds1d_mixup_ema_200ep_es.json"
FINETUNE_CONFIG="configs/server3090_kv260_ds1d_pyramid_finetune_kdprotect_200ep_es.json"

BASE_EXP="server3090_full10_pyramid_base_200ep_es"
AVGMAX_EXP="server3090_full10_avgmax_mixup_200ep_es"
FINETUNE_EXP="server3090_full10_pyramid_finetune_kdprotect_200ep_es"

for path in "$BASE_CONFIG" "$AVGMAX_CONFIG" "$FINETUNE_CONFIG" tools/run_multifold.py train.py; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

echo "=== Server3090 full-10 runner ==="
echo "repo: $(pwd)"
echo "python: $PYTHON_BIN"
echo "folds: $FOLDS"
echo "epochs: $EPOCHS"
echo "batch_size_override: ${BATCH_SIZE:-none}"
echo "skip_existing: $SKIP_EXISTING"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo

"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

parse_folds() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
value = sys.argv[1]
folds = []
for part in value.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        start, end = part.split("-", 1)
        folds.extend(range(int(start), int(end) + 1))
    else:
        folds.append(int(part))
for fold in dict.fromkeys(folds):
    if not 1 <= fold <= 10:
        raise SystemExit(f"fold must be in [1, 10], got {fold}")
    print(fold)
PY
}

mapfile -t FOLD_LIST < <(parse_folds "$FOLDS")

run_experiment() {
  local name="$1"
  local config="$2"
  local exp_name="$3"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$LOG_ROOT/${timestamp}_${exp_name}.log"

  local cmd=(
    "$PYTHON_BIN" tools/run_multifold.py
    --config "$config"
    --exp_name "$exp_name"
    --folds "$FOLDS"
    --epochs "$EPOCHS"
    --analyze
    --eval_modes
  )
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip_existing)
  fi

  echo
  echo "=== START: $name ==="
  echo "time: $(date -Is)"
  echo "log: $log_file"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  echo

  "${cmd[@]}" 2>&1 | tee "$log_file"

  echo "=== DONE: $name ==="
  echo "time: $(date -Is)"
}

check_baseline_checkpoints() {
  local missing=0
  for fold in "${FOLD_LIST[@]}"; do
    local ckpt="experiments/$BASE_EXP/fold_${fold}/checkpoints/tcam_fold_${fold}_cycle_final.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "Missing baseline checkpoint required for fine-tune: $ckpt" >&2
      missing=1
    fi
  done
  if [[ "$missing" != "0" ]]; then
    echo "Run failed before protected fine-tune because baseline checkpoints are incomplete." >&2
    exit 1
  fi
}

run_experiment "1/3 current best pyramid baseline" "$BASE_CONFIG" "$BASE_EXP"
run_experiment "2/3 lightweight avgmax mixup EMA" "$AVGMAX_CONFIG" "$AVGMAX_EXP"
check_baseline_checkpoints
run_experiment "3/3 protected fine-tune from pyramid baseline" "$FINETUNE_CONFIG" "$FINETUNE_EXP"

echo
echo "=== ALL REQUESTED RUNS COMPLETED ==="
echo "Summaries:"
echo "  experiments/$BASE_EXP/multifold_summary.md"
echo "  experiments/$AVGMAX_EXP/multifold_summary.md"
echo "  experiments/$FINETUNE_EXP/multifold_summary.md"
