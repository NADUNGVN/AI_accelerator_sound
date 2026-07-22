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

## 4. Results so far + v2 recipe (target >90%)

### 4.0 MVP v1 outcome (do not reuse as best teacher)

| Run | Best val | Best-val test | Note |
|-----|---------:|--------------:|------|
| Local 12ep | 89.26% | **89.89%** | Still best teacher reference |
| **MVP v1 RTX8000** `...mvp...24ep` | 88.45% @ep3 | **88.28%** | ES early; ep7–8 test ~89.5% but val lower |

MVP problems: early-stop on val@3, weighted_sampler, freeze=1 / lr khác local.

### 4.1 Locked split (unchanged)

```text
protocol = source_group_8_1_1
seed     = 83
fold     = 1
model    = MIT/ast-finetuned-audioset-10-10-0.4593
```

### 4.2 Teacher v2 recipe (push toward 90–92%+)

| Knob | Local ~90% | MVP v1 (fail) | **v2 (now)** |
|------|------------|---------------|--------------|
| epochs | 12 | 24 (ES@9) | **30** |
| ES | soft | warmup 8 pat 6 | **warmup 12 pat 12** (patient) |
| weighted_sampler | off | **on** | **off** |
| freeze_base | 2 | 1 | **2** |
| encoder_lr / head_lr | 1e-5 / 5e-4 | 1.5e-5 / 3e-4 | **1e-5 / 5e-4** |
| batch × accum | 4×4 | 8×2 | **12×2** (eff 24, 48GB) |
| num_workers | 0 | 8 | **10** |

**Realistic bar:** best-val test **> 89.89%** (beat local).  
**Stretch:** **≥ 91%**.  
**Hard ceiling note:** **95%** on SDP fold1 is optimistic with this AST; aim step-wise 90→92 first. engine_idling / fsID 144007 remains the main wall.

```bash
# after git pull e202ddb+ with v2 script
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sound_ast
cd $HOME/Dung_TDTU/AI_accelerator_sound
screen -S ast_v2
bash scripts/run_ast_teacher_mvp_sdp811.sh
# Ctrl+A D to detach
```

Exp folder: `experiments/server8000_ast_teacher_v2_sdp811_f1_30ep/`

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
