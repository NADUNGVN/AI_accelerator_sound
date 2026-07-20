#!/usr/bin/env bash
# Run Abdoli et al. 2019 paper setup over official UrbanSound8K 10 folds.
#
# Usage (from repo root):
#   bash scripts/run_paper_abdoli_10fold.sh
#   bash scripts/run_paper_abdoli_10fold.sh --start_fold 2 --end_fold 10
#   bash scripts/run_paper_abdoli_10fold.sh --config configs/paper_abdoli_rand.json --exp_name paper_abdoli_rand
#
# After all folds finish:
#   python scripts/summarize_10fold.py --exp_name paper_abdoli_gamma

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/paper_abdoli_gamma.json"
EXP_NAME="paper_abdoli_gamma"
DATA_DIR="data/raw/UrbanSound8K"
START_FOLD=1
END_FOLD=10
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --start_fold) START_FOLD="$2"; shift 2 ;;
    --end_fold) END_FOLD="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/metadata/UrbanSound8K.csv" ]]; then
  echo "Dataset not found under: $DATA_DIR" >&2
  echo "Expected: $DATA_DIR/metadata/UrbanSound8K.csv" >&2
  exit 1
fi

LOG_DIR="experiments/${EXP_NAME}/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/10fold_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$MASTER_LOG"
echo "Abdoli 10-fold CV" | tee -a "$MASTER_LOG"
echo "  config   : $CONFIG" | tee -a "$MASTER_LOG"
echo "  exp_name : $EXP_NAME" | tee -a "$MASTER_LOG"
echo "  data_dir : $DATA_DIR" | tee -a "$MASTER_LOG"
echo "  folds    : ${START_FOLD}..${END_FOLD}" | tee -a "$MASTER_LOG"
echo "  python   : $PYTHON" | tee -a "$MASTER_LOG"
echo "  log      : $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

FAILED=0
for f in $(seq "$START_FOLD" "$END_FOLD"); do
  FOLD_LOG="${LOG_DIR}/fold_${f}.log"
  echo "" | tee -a "$MASTER_LOG"
  echo "-------- FOLD ${f}/${END_FOLD}  $(date -Is) --------" | tee -a "$MASTER_LOG"

  # Skip completed folds (metrics.json already present)
  METRICS="experiments/${EXP_NAME}/fold_${f}/metrics.json"
  if [[ -f "$METRICS" ]]; then
    echo "[skip] fold ${f}: $METRICS already exists" | tee -a "$MASTER_LOG"
    continue
  fi

  set +e
  "$PYTHON" train.py \
    --fold "$f" \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --exp_name "$EXP_NAME" \
    2>&1 | tee "$FOLD_LOG" | tee -a "$MASTER_LOG"
  RC=${PIPESTATUS[0]}
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "[FAIL] fold ${f} exit code $RC" | tee -a "$MASTER_LOG"
    FAILED=1
    # continue remaining folds; comment next line to abort on first failure
    # exit $RC
  else
    echo "[OK] fold ${f} finished" | tee -a "$MASTER_LOG"
  fi
done

echo "" | tee -a "$MASTER_LOG"
echo "======== 10-fold run finished  $(date -Is)  failed_flag=$FAILED ========" | tee -a "$MASTER_LOG"

"$PYTHON" scripts/summarize_10fold.py --exp_name "$EXP_NAME" | tee -a "$MASTER_LOG"

exit "$FAILED"
