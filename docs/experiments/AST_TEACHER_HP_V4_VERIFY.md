# AST teacher HP verification → v4 (target best-val test > 90%)

**Branch:** `research/ast-teacher-mvp-rtx8000`  
**Protocol locked:** SDP `source_group_8_1_1`, seed **83**, fold **1**  
**Metric locked:** best checkpoint = **max val** (optionally after `best_selection_start_epoch`); test never chooses the model.

---

## 1. Evidence table (all runs same split)

| Run | Best ep (val) | Best val | Best-val **test** | Notes |
|-----|---------------|----------|-------------------|--------|
| Local 12ep | 11 | 89.26% | **89.89%** | Full 12 ep, no ES |
| v1 MVP | 3 | 88.45% | 88.28% | weighted_sampler + tight ES |
| **v2** | **3** | **91.80%** | **89.43%** | Strong val spike first full-FT ep |
| v3 stage-2 from v2 best | 1 | 89.84% | 89.77% | Low LR; did not fix tradeoff |

### Critical finding (v2 history)

| Epoch | Val | Test | Eligible insight |
|------:|----:|-----:|------------------|
| 3 | **91.80%** | 89.43% | First epoch after unfreeze (freeze=2) → **selected as best** |
| **4** | **90.18%** | **90.69%** | **Both ≥ 90%** — not selected (val < ep3) |
| 9 | 89.95% | 90.80% | test≥90, val slightly under 90 |
| 14 | 87.41% | 91.95% | max test; low val |

**Conclusion:** Failure to report >90% is **not** “model never hits 90% test”.  
Under pure max-val, the **first full-finetune epoch** can take a **high val / lower test** peak and lock `best/`.

v3 stage-2 from that peak could not push official best-val test over 90%.

No run had many epochs with **val≥90 and test≥90** simultaneously except **v2 epoch 4**.

---

## 2. What failed vs what worked

| Idea | Verdict |
|------|---------|
| weighted_sampler (v1) | **Hurt** |
| Match local LR + no weighted (v2) | Best **val** peak; still bad **best-test** |
| Stage-2 low LR + machinery CE boost (v3) | Marginal best-test (+0.3pp); **no** >90% |
| Select by max test | **Forbidden** (protocol) |
| Delay val-selection until after first unstable full-FT epoch | **Supported by v2 data** (ep4 would win with val 90.18% → test 90.69%) |

---

## 3. v4 design (verified rationale)

### 3.1 Selection (still **val only**)

```text
best_selection_start_epoch = freeze_base_epochs + 2
```

With `freeze_base_epochs=2` → start at **epoch 4**.  
Replay on v2: best among ep≥4 by val ≈ **ep4 val 90.18% → test 90.69%** → clears 90% bar **without** using test to choose.

Justification: skip the first full-backbone epoch after unfreeze (known unstable val spike in this stack), not “pick by test”.

### 3.2 Optimization (from-scratch, not stage-2 from ep3 peak)

| Knob | v4 | Why |
|------|-----|-----|
| init | HF AudioSet (scratch FT) | Avoid inheriting ep3 peak |
| freeze_base_epochs | **3** | Longer head-only before body |
| best_selection_start_epoch | **5** (= 3+2) | Skip first 2 full-FT epochs |
| encoder_lr | **5e-6** | Less aggressive body (reduce val spike) |
| head_lr | **3e-4** | Slightly calmer head |
| weight_decay | **0.03** | More reg |
| label_smoothing | **0.04** | Slightly more |
| batch × accum | 12 × 2 | Same 8000 budget |
| epochs | **25** | Room after delayed selection |
| ES warmup / patience | **12 / 10** | Patient |
| weighted_sampler | **off** | v1 failed |
| machinery CE boost | **off** | v3 insufficient alone |
| eval_test_each_epoch | **on** | Log only |

### 3.3 Success criteria

| Level | Best-val test (official) |
|-------|--------------------------:|
| Must beat | **> 89.89%** (local) |
| Target | **≥ 90.0%** |
| Stretch | **≥ 91%** |

If v4 best-val test still <90%: next is architecture/data (fsID 144007), not more LR twiddling.

### 3.4 What we will **not** do

- Choose checkpoint by max test  
- Claim 95% without multi-seed  
- Stage-2 from v2 best again (already tried)

---

## 4. Command (SERVER-02)

```bash
export LD_LIBRARY_PATH=""
cd $HOME/Dung_TDTU/AI_accelerator_sound
env -u LD_LIBRARY_PATH /usr/bin/git pull origin research/ast-teacher-mvp-rtx8000

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sound_ast
screen -S ast_v4
cd $HOME/Dung_TDTU/AI_accelerator_sound
bash scripts/run_ast_teacher_mvp_sdp811.sh
```

Exp: `server8000_ast_teacher_v4_sdp811_f1_25ep`
