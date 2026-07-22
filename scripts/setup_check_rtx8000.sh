#!/usr/bin/env bash
# One-shot sanity check on SERVER-02 RTX 8000 after clone + data
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== host ==="
hostname || true
uname -a || true

echo "=== GPU ==="
nvidia-smi || { echo "nvidia-smi missing"; exit 1; }

echo "=== python / torch ==="
python - <<'PY'
import sys
print("python", sys.version)
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("transformers FAIL", e)
    raise
PY

echo "=== data ==="
DATA_DIR="${DATA_DIR:-data/UrbanSound8K}"
test -f "$DATA_DIR/metadata/UrbanSound8K.csv" && echo "OK metadata" || { echo "MISSING $DATA_DIR/metadata/UrbanSound8K.csv"; exit 2; }
n_wav=$(find "$DATA_DIR/audio" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
echo "wav_count=$n_wav (expect 8732)"
test "$n_wav" = "8732" && echo "OK wav count" || echo "WARN wav count != 8732"

echo "=== SDP fingerprint ==="
python tools/verify_sdp_split_fingerprint.py --data_dir "$DATA_DIR" --fold 1 --seed 83 --protocol source_group_8_1_1

echo "=== tools ==="
test -f tools/finetune_ast_teacher.py && echo "OK finetune_ast_teacher.py"
test -f tools/cache_ast_teacher_logits.py && echo "OK cache_ast_teacher_logits.py"

echo "ALL CHECKS DONE"
