#!/usr/bin/env bash
set -Eeuo pipefail

DATASET="all"
DATA_ROOT="${DATA_ROOT:-data/raw}"

ESC50_URL="${ESC50_URL:-https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip}"
SPEECH_URL="${SPEECH_URL:-http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz}"

usage() {
  cat <<'USAGE'
Download and verify Phase 1 datasets.

Usage:
  bash scripts/download_phase1_datasets.sh [options]

Options:
  --dataset NAME      all | esc50 | speech_commands. Default: all
  --data-root PATH    Dataset root. Default: data/raw
  -h, --help          Show this help.

Expected output layout:
  data/raw/ESC-50/
    meta/esc50.csv
    audio/*.wav

  data/raw/speech_commands_v0.02/
    validation_list.txt
    testing_list.txt
    _background_noise_/
    <word>/*.wav
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
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

case "$DATASET" in
  all|esc50|speech_commands) ;;
  *)
    echo "Unsupported --dataset '$DATASET'. Use all, esc50, or speech_commands." >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")/.."
mkdir -p "$DATA_ROOT"

download_file() {
  local url="$1"
  local out="$2"

  if [[ -s "$out" ]]; then
    echo "Reusing existing download: $out"
    return
  fi

  echo "Downloading: $url"
  echo "Output     : $out"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --continue-at - "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$out"
  else
    echo "Need curl or wget to download datasets." >&2
    exit 1
  fi
}

extract_zip_with_python() {
  local archive="$1"
  local out_dir="$2"
  python - "$archive" "$out_dir" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out_dir)
PY
}

verify_esc50() {
  local target="$1"
  if [[ ! -f "$target/meta/esc50.csv" || ! -d "$target/audio" ]]; then
    echo "ESC-50 verification failed: expected meta/esc50.csv and audio/ under $target" >&2
    exit 1
  fi
  local wav_count
  wav_count="$(find "$target/audio" -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')"
  if [[ "$wav_count" != "2000" ]]; then
    echo "ESC-50 verification failed: expected 2000 wav files, found $wav_count" >&2
    exit 1
  fi
  echo "ESC-50 ready: $target ($wav_count wav files)"
}

download_esc50() {
  local target="$DATA_ROOT/ESC-50"
  if [[ -f "$target/meta/esc50.csv" && -d "$target/audio" ]]; then
    verify_esc50 "$target"
    return
  fi

  local archive="$DATA_ROOT/ESC-50-master.zip"
  local extract_root="$DATA_ROOT/_esc50_extract"
  rm -rf "$extract_root"
  mkdir -p "$extract_root"

  download_file "$ESC50_URL" "$archive"
  extract_zip_with_python "$archive" "$extract_root"

  local extracted
  extracted="$(find "$extract_root" -maxdepth 4 -type f -path '*/meta/esc50.csv' -print -quit)"
  if [[ -z "$extracted" ]]; then
    echo "ESC-50 archive extraction did not contain meta/esc50.csv" >&2
    exit 1
  fi
  extracted="$(dirname "$(dirname "$extracted")")"

  rm -rf "$target"
  mv "$extracted" "$target"
  rm -rf "$extract_root"
  verify_esc50 "$target"
}

verify_speech_commands() {
  local target="$1"
  if [[ ! -f "$target/validation_list.txt" || ! -f "$target/testing_list.txt" || ! -d "$target/_background_noise_" ]]; then
    echo "Speech Commands verification failed: expected validation_list.txt, testing_list.txt, and _background_noise_/ under $target" >&2
    exit 1
  fi
  local wav_count
  wav_count="$(find "$target" -type f -name '*.wav' | wc -l | tr -d ' ')"
  if [[ "$wav_count" -lt "100000" ]]; then
    echo "Speech Commands verification failed: expected at least 100000 wav files, found $wav_count" >&2
    exit 1
  fi
  echo "Speech Commands ready: $target ($wav_count wav files)"
}

download_speech_commands() {
  local target="$DATA_ROOT/speech_commands_v0.02"
  if [[ -f "$target/validation_list.txt" && -f "$target/testing_list.txt" && -d "$target/_background_noise_" ]]; then
    verify_speech_commands "$target"
    return
  fi

  local archive="$DATA_ROOT/speech_commands_v0.02.tar.gz"
  mkdir -p "$target"

  download_file "$SPEECH_URL" "$archive"
  tar -xzf "$archive" -C "$target"
  verify_speech_commands "$target"
}

if [[ "$DATASET" == "all" || "$DATASET" == "esc50" ]]; then
  download_esc50
fi

if [[ "$DATASET" == "all" || "$DATASET" == "speech_commands" ]]; then
  download_speech_commands
fi
