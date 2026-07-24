#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON:-python}"
DATASET="all"
STAGE="smoke"
SKIP_EXISTING="1"
ANALYZE="1"
EVAL_MODES="1"
BATCH_SIZE=""
ESC50_DATA="${ESC50_DATA:-}"
SPEECH_DATA="${SPEECH_COMMANDS_DATA:-${SPEECH_DATA:-}}"

ESC50_CONFIG="configs/main/student_ds_conv2d_h1_pyramid_esc50_phase1.json"
SPEECH_CONFIG="configs/main/student_ds_conv2d_h1_pyramid_speech_commands_phase1.json"

usage() {
  cat <<'USAGE'
Run Phase 1 multi-dataset training for ESC-50 and Speech Commands.

Usage:
  bash scripts/run_phase1_multidataset.sh [options]

Options:
  --stage NAME          smoke | full-first | full-all. Default: smoke
  --dataset NAME        all | esc50 | speech_commands. Default: all
  --python PATH         Python executable. Default: python
  --batch-size N        Override config batch_size.
  --esc50-data PATH     Override ESC-50 root. Expected: meta/esc50.csv + audio/
  --speech-data PATH    Override Speech Commands v0.02 root.
  --no-skip-existing    Re-run even if metrics/analysis already exist.
  --no-analyze          Do not run tools/analyze_experiment.py after train.
  --no-eval-modes       Do not run extra eval aggregation modes.
  -h, --help            Show this help.

Examples:
  # Gate 1: tiny smoke for both datasets.
  bash scripts/run_phase1_multidataset.sh --stage smoke

  # Gate 2: first canonical train split for both datasets.
  bash scripts/run_phase1_multidataset.sh --stage full-first

  # ESC-50 all five validation-selected test folds + Speech official split.
  bash scripts/run_phase1_multidataset.sh --stage full-all
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --esc50-data)
      ESC50_DATA="$2"
      shift 2
      ;;
    --speech-data)
      SPEECH_DATA="$2"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING="0"
      shift
      ;;
    --no-analyze)
      ANALYZE="0"
      shift
      ;;
    --no-eval-modes)
      EVAL_MODES="0"
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

case "$STAGE" in
  smoke|full-first|full-all) ;;
  *)
    echo "Unsupported --stage '$STAGE'. Use smoke, full-first, or full-all." >&2
    exit 2
    ;;
esac

case "$DATASET" in
  all|esc50|speech_commands) ;;
  *)
    echo "Unsupported --dataset '$DATASET'. Use all, esc50, or speech_commands." >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "$ESC50_CONFIG" "$SPEECH_CONFIG" tools/run_multifold.py train.py tools/analyze_experiment.py; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

echo "=== Phase 1 multi-dataset runner ==="
echo "repo: $(pwd)"
echo "python: $PYTHON_BIN"
echo "stage: $STAGE"
echo "dataset: $DATASET"
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

resolve_data_dir() {
  local dataset="$1"
  local override="$2"
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
    return
  fi
  "$PYTHON_BIN" - "$dataset" <<'PY'
import sys
from train import default_data_dir
print(default_data_dir(sys.argv[1]))
PY
}

check_esc50_data() {
  local data_dir="$1"
  if [[ ! -f "$data_dir/meta/esc50.csv" || ! -d "$data_dir/audio" ]]; then
    echo "ESC-50 data not ready: $data_dir" >&2
    echo "Expected: $data_dir/meta/esc50.csv and $data_dir/audio/" >&2
    exit 1
  fi
}

check_speech_data() {
  local data_dir="$1"
  if [[ ! -f "$data_dir/validation_list.txt" || ! -f "$data_dir/testing_list.txt" || ! -d "$data_dir/_background_noise_" ]]; then
    echo "Speech Commands v0.02 data not ready: $data_dir" >&2
    echo "Expected: validation_list.txt, testing_list.txt, and _background_noise_/" >&2
    exit 1
  fi
}

run_multifold() {
  local config="$1"
  local exp_name="$2"
  local folds="$3"
  local data_dir="$4"
  shift 4

  local command=(
    "$PYTHON_BIN" tools/run_multifold.py
    --config "$config"
    --exp_name "$exp_name"
    --folds "$folds"
    --data_dir "$data_dir"
  )

  if [[ -n "$BATCH_SIZE" ]]; then
    command+=(--batch_size "$BATCH_SIZE")
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    command+=(--skip_existing)
  fi
  if [[ "$ANALYZE" == "1" ]]; then
    command+=(--analyze)
  fi
  if [[ "$EVAL_MODES" == "1" ]]; then
    command+=(--eval_modes)
  fi

  command+=("$@")

  echo
  printf '>>>'
  printf ' %q' "${command[@]}"
  printf '\n'
  "${command[@]}"
}

run_esc50() {
  local data_dir
  data_dir="$(resolve_data_dir esc50 "$ESC50_DATA")"
  check_esc50_data "$data_dir"
  echo "ESC-50 data: $data_dir"

  case "$STAGE" in
    smoke)
      run_multifold "$ESC50_CONFIG" "esc50_phase1_fold1_smoke_1ep" "1" "$data_dir" \
        --epochs 1 --max_train_clips 64 --max_val_clips 32 --max_test_clips 32
      ;;
    full-first)
      run_multifold "$ESC50_CONFIG" "esc50_phase1_dsconv2dh1_fold1_50ep" "1" "$data_dir"
      ;;
    full-all)
      run_multifold "$ESC50_CONFIG" "esc50_phase1_dsconv2dh1_5fold_50ep" "1-5" "$data_dir"
      ;;
  esac
}

run_speech_commands() {
  local data_dir
  data_dir="$(resolve_data_dir speech_commands "$SPEECH_DATA")"
  check_speech_data "$data_dir"
  echo "Speech Commands data: $data_dir"

  case "$STAGE" in
    smoke)
      run_multifold "$SPEECH_CONFIG" "speech_commands_phase1_official12_smoke_1ep" "1" "$data_dir" \
        --epochs 1 --max_train_clips 128 --max_val_clips 64 --max_test_clips 64
      ;;
    full-first|full-all)
      run_multifold "$SPEECH_CONFIG" "speech_commands_phase1_official12_dsconv2dh1_30ep" "1" "$data_dir"
      ;;
  esac
}

if [[ "$DATASET" == "all" || "$DATASET" == "esc50" ]]; then
  run_esc50
fi

if [[ "$DATASET" == "all" || "$DATASET" == "speech_commands" ]]; then
  run_speech_commands
fi
