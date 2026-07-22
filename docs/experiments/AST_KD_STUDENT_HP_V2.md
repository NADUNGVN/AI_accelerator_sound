# Student AST-KD HP v2 (after 72.6% failure)

## Diagnosis

| Run | Init | KD | best-val test |
|-----|------|-----|--------------:|
| Model B self-KD | strong DS1D ~79% | same-arch teacher λ=0.6 | **80.00%** |
| AST-KD v1 (8000) | **from scratch** | AST cache λ=0.5 T=2, no mixup-KD | **72.64%** |
| Prior AST-KD fold1 local | from scratch-ish | AST | ~75% |

**Root causes of 72%:**

1. **No strong student init** — Model B wins because it fine-tunes a ~79% model; AST-KD v1 learned CE+KD from zero under hard soft-label domain gap (spectrogram AST → waveform CNN).  
2. **KD too hard / always on from scratch** — soft labels from ~90% teacher but different features can dominate early training.  
3. **`apply_to_mixup=false`** — many batches only CE; KD signal inconsistent.  
4. Teacher soft labels still wrong on engine_idling cluster — KD can teach systematic errors if λ high.

## v2 method (two-phase, same SDP f1 seed 83)

### Phase A — No-Teacher student baseline on 8000

- Config: `configs/student_ds1d_noteacher_sdp811_server_val.json`  
- Exp: `server8000_student_noteacher_sdp811_f1_50ep`  
- Goal: solid init (~76–79% band).  
- Checkpoint: `.../fold_1/checkpoints/tcam_fold_1_best.pt`

### Phase B — Fine-tune + AST-v3 cached KD

- Config: `configs/student_ds1d_ast_v3_kd_finetune_v2_sdp811_val.json`  
- Exp: `server8000_student_ast_v3_kd_ft_v2_sdp811_f1_30ep`  
- Init: Phase A **best.pt**  
- KD: same `ast_teacher_logits_v3_sdp811` cache (teacher T* v3)  

| Knob | AST-KD v1 (72%) | **v2 finetune** | Why |
|------|-----------------|-----------------|-----|
| init | scratch | **no-teacher best** | Match Model B pattern |
| lr | 1e-3 | **2e-4** | Fine-tune, not relearn |
| epochs | 50 | **30** + ES | |
| KD weight λ | 0.50 | **0.35** | Softer teacher pressure |
| temperature T | 2.0 | **2.5** | Softer distribution |
| apply_to_mixup | false | **true** | KD under mixup too |
| mixup prob | 0.50 | **0.40** | Slightly more pure CE/KD batches |
| protect_classes | [] | **[0,4,5,7]** machinery | Focus KD where student fails |
| label_smoothing | 0.02 | 0.02 | Keep mild with KD |

**Success bar:** best-val test **> 72.64%** (must); stretch **> 80%** (beat Model B).  
**Fail:** ≤75% → reduce λ further (0.2) or stop AST-KD as main student path.

## Commands (SERVER-02)

```bash
# pull
export LD_LIBRARY_PATH=""
cd $HOME/Dung_TDTU/AI_accelerator_sound
env -u LD_LIBRARY_PATH /usr/bin/git pull origin research/ast-teacher-mvp-rtx8000

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sound_ast

# Phase A
python tools/run_multifold.py \
  --config configs/student_ds1d_noteacher_sdp811_server_val.json \
  --exp_name server8000_student_noteacher_sdp811_f1_50ep \
  --folds 1 --epochs 50 \
  --data_dir data/UrbanSound8K \
  --analyze --eval_modes

# confirm init exists
ls experiments/server8000_student_noteacher_sdp811_f1_50ep/fold_1/checkpoints/tcam_fold_1_best.pt

# Phase B (reuse existing v3 logits cache)
ls experiments/ast_teacher_logits_v3_sdp811/fold_1/teacher_logits.pt

python tools/run_multifold.py \
  --config configs/student_ds1d_ast_v3_kd_finetune_v2_sdp811_val.json \
  --exp_name server8000_student_ast_v3_kd_ft_v2_sdp811_f1_30ep \
  --folds 1 --epochs 30 \
  --data_dir data/UrbanSound8K \
  --analyze --eval_modes
```

If logits cache missing, re-run cache from teacher v3 best (see AST_V3_DISTILL_RUNBOOK.md).
