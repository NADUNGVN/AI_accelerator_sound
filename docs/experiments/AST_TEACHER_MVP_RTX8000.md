# AST Teacher MVP — SERVER-02 RTX 8000 (SDP 8-1-1)

**Branch:** `research/ast-teacher-mvp-rtx8000`  
**Machine:** SERVER-02 · **Quadro RTX 8000 48 GB** · ~188 GB RAM  
**Goal:** retrain AST teacher under the **same SDP seed contract** as student Model A/B, aim higher fold-1 test than current **~90%**, then (later) distill to student.

---

## 1. What the previous teacher used (verified)

From `experiments/local_ast_teacher_finetune_f1_12ep/fold_1/metrics.json` and fold2 deferred:

| Field | Prior AST fine-tune |
|-------|---------------------|
| **protocol** | **`source_group_8_1_1`** (= SDP 8-1-1) |
| **seed** | **83** |
| **model** | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| **fold 1** | best-val test **89.89%**, final test **90.23%**, best val **89.26%**, 12 ep |
| **fold 2** | best-val test **85.33%**, early-stop @6 |
| **train↔test fsID+label overlap** | **0** |
| **train↔val overlap** | **0** |

Script defaults (`tools/finetune_ast_teacher.py`):

```text
--protocol source_group_8_1_1
--seed 83
```

Student Model A/B fold1 (same protocol/seed):

```text
train=6996  val=866  test=870  overlap train/test = 0
```

**Conclusion:** prior teacher already used **SDP 8-1-1 + seed 83**. This MVP **keeps that contract** (do not invent a new split for “higher score”).  
Optional later: *extra sampling weights inside train only* (same train pool), not a new protocol.

### Re-verify split on server (mandatory first step)

```bash
python tools/verify_sdp_split_fingerprint.py --fold 1 --seed 83 --protocol source_group_8_1_1
# expect train/val/test counts 6996/866/870 and overlap 0
```

---

## 2. What to do now (order)

| Step | Where | Action |
|------|--------|--------|
| 1 | Laptop | Branch + push (this doc + tools) |
| 2 | SERVER-02 | Clone/pull branch, Python env, CUDA, HF cache |
| 3 | SERVER-02 | Place **UrbanSound8K** under `data/UrbanSound8K` |
| 4 | SERVER-02 | Run split fingerprint verify |
| 5 | SERVER-02 | Smoke 1 epoch AST (tiny) |
| 6 | SERVER-02 | Full MVP teacher fold1 |
| 7 | Either | Push `metrics.json` + `summary.md` (not full weights unless asked) |

**Do not** start student KD until teacher fold1 best-val test is recorded and (ideally) ≥ prior 89.89%.

---

## 3. Server setup (RTX 8000) — first time

```bash
# example paths — adjust user/home
hostname   # should be SERVER-02 (or your lab name)
nvidia-smi # Quadro RTX 8000 / 48GB

cd ~
mkdir -p Dung_TDTU
cd Dung_TDTU
git clone https://github.com/NADUNGVN/AI_accelerator_sound.git
cd AI_accelerator_sound
git fetch origin
git checkout research/ast-teacher-mvp-rtx8000
git pull origin research/ast-teacher-mvp-rtx8000

# env (create once)
conda create -n sound_ast python=3.11 -y
conda activate sound_ast
pip install -r requirements.txt
pip install transformers accelerate datasets soundfile librosa

# data (choose one)
mkdir -p data
# scp / rsync UrbanSound8K_on_server.tar.gz then:
# tar -xzf UrbanSound8K_on_server.tar.gz -C data
# expect: data/UrbanSound8K/metadata/UrbanSound8K.csv + audio/fold1..10

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=$PWD/experiments/hf_cache
mkdir -p "$HF_HOME"

python - <<'PY'
import torch
print("cuda", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("vram_gb", torch.cuda.get_device_properties(0).total_memory/1024**3)
PY
```

Helper script: `scripts/setup_check_rtx8000.sh` (after clone).

---

## 4. MVP train recipe (higher accuracy attempt)

**Locked (must not change without renaming protocol):**

```text
protocol = source_group_8_1_1
seed     = 83
fold     = 1   # first MVP
model    = MIT/ast-finetuned-audioset-10-10-0.4593
```

**Changed vs prior local 12ep (use 48 GB):**

| Knob | Prior (local) | MVP RTX 8000 |
|------|---------------|--------------|
| epochs | 12 | **24** (ES patience 6 after warmup 8) |
| batch | 4 × accum 4 | **8 × accum 2** (eff 16) or **16 × accum 1** if VRAM OK |
| freeze_base_epochs | 2 | **1** (unfreeze encoder earlier) |
| encoder_lr | 1e-5 | **1.5e-5** |
| head_lr | 5e-4 | **3e-4** |
| weighted_sampler | off | **on** (AST weak: engine_idling / machinery) |
| num_workers | 0 | **8** |
| eval_test_each_epoch | optional | **on** (log only; still select by **val**) |

**Sampling note (same train set, reweight only):**  
AST fold1 per-class showed **engine_idling ~66%** (jackhammer confusions). Weighted sampler + balanced CE targets that *inside* SDP train — **does not** move clips across train/val/test.

```bash
screen -S ast_teacher_mvp

python tools/finetune_ast_teacher.py \
  --data_dir data/UrbanSound8K \
  --exp_name server8000_ast_teacher_mvp_sdp811_f1_24ep \
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
  --hf_cache_dir experiments/hf_cache \
  --device cuda
```

Or: `bash scripts/run_ast_teacher_mvp_sdp811.sh`

**Primary metric:** `best_test` at **best val** epoch (same as before).  
**Success bar (MVP):** beat **89.89%** best-val test on fold1; stretch **≥91%**.  
**Fail bar:** ≤ prior without clear bug → do not claim improvement; analyze weak sources first.

---

## 5. After train

```bash
# light metrics only
git add -f \
  experiments/server8000_ast_teacher_mvp_sdp811_f1_24ep/fold_1/metrics.json \
  experiments/server8000_ast_teacher_mvp_sdp811_f1_24ep/fold_1/summary.md
git commit -m "Add AST teacher MVP fold1 metrics (SDP 8-1-1 seed 83) on RTX 8000"
git push origin research/ast-teacher-mvp-rtx8000
```

Keep **checkpoints** on server disk (large). Path example:

```text
experiments/server8000_ast_teacher_mvp_sdp811_f1_24ep/fold_1/checkpoints/best
```

Next phase (not this MVP): `tools/cache_ast_teacher_logits.py` → student KD redesign.

---

## 6. What not to do

- Change protocol/seed “to get higher number” without documenting a new experiment name  
- Select teacher checkpoint by **test**  
- Train student on SERVER-02 before teacher MVP is logged  
- Mix nighttime-tsd / other projects env with this repo blindly  
