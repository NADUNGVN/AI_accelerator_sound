# Achieved (main truth — no local vs server split)

**Rule:** One research stack. Report the **best verified run** for each goal.  
Machine (laptop / 3090) only matters for *where files live*, not for *which number is official research*.

**Main path stack (in git `main`)**

| Item | Value |
|---|---|
| Model | **DS-Conv2D-H1 Pyramid** (`ds_conv2d_h1_pyramid`; legacy `kv260_audio_net_ds1d`) full-clip |
| Params / MAC | **101 674** / **61 854 400** |
| Config (no-teacher) | `configs/main/student_ds_conv2d_h1_pyramid_sourcegroup.json` (= `configs/kv260_ds1d_pyramid_mixup_ema_val.json`) |
| Protocol | `source_group_8_1_1`, seed **83**, train/val/test, best-val checkpoint |
| Primary single metric | `test_acc_best_val_model` |
| Ensemble metric | `test_acc_ensemble` (last-2) |

---

## Three tracks — target vs **best achieved**

| Track | Target | **Best achieved** | Exp (canonical evidence) | Status vs 80–85% |
|---|---|---:|---|---|
| **1. Single model** | 80–85% | **79.08%** best-val test (**fold1 peak**) | `local_multifold_pyramid_base_f1_f3_50ep` / fold_1 | **Below 80%**; see variance note |
| **2. Ensemble** | 80–85% | **79.89%** ensemble (**fold1 peak**) | same exp fold_1 | **Near 80%** on that fold only |
| **3. KD student** | 80–85% | **80.00%** best-val test / **80.23%** ens | `local_finetune_kdprotect_f1_20ep` / fold_1 | **Hits 80% band** (not yet 85%) |

### Track 3 teacher clarity (do not mix)

| Role | What it is | Accuracy (SDP fold 1 unless noted) |
|------|------------|-------------------------------------|
| **Model B student (deliverable)** | DS-Conv2D-H1 after **KD-protect** fine-tune | **80.00%** bvt / **80.23%** ens |
| **Teacher of that 80% run** | **Same-family** DS1D `cycle_final` (not AST) | ~**79%** lineage (Model A best **79.08%**) |
| **AST teacher (research only)** | Fine-tuned `MIT/ast-finetuned-audioset-…` | best-val test **89.89%**; final test **90.23%** |
| **AST train-cache (fold 2 doc)** | Soft-label cache quality on **train** | ~**92.4%** — **not** deploy test |
| **AST→student KD (fold 1)** | `local_fold1_kv260_ast_teacher_kd_50ep` | student ~**75.4%** — **weaker** than Model B |

Full narrative: [`docs/paper/MODEL_B_KD.md`](../paper/MODEL_B_KD.md).

### Variance note (79% is not “guaranteed recipe output”)

Same MAIN-family config `kv260_ds1d_pyramid_mixup_ema_val.json`, seed **83**, no-teacher:

| Evidence | best-val test |
|---|---:|
| Fold1 “79.08%” files | **79.08%** — **two paths, identical metrics** → **1 run duplicated**, not 2 independent seeds |
| Same multifold exp **fold2** | **67.67%** |
| Same multifold exp **fold3** | **66.93%** |
| Folds 1–3 **mean** bvt | **~71.2%** (std ~5.6) |
| 3090 MAIN 50ep refresh | **77.70%** |
| 3090 notacher long run | **76.90%** |
| Local same cfg fold1 **200ep** | **76.21%** |
| 3090 100ep+ES | **67.82%** (reject) |

**Interpretation:** 79.08% / 79.89% = **fold1 high-water mark** for this seed/recipe, **not** a multi-fold or multi-seed mean.  
**Working 3090 fold1 band** for MAIN 50ep: about **76.9–77.7%** single.  
Claiming “the model is 79%” without saying **fold1** overstates stability.

**Do not write:** “Track 3 teacher = 92% AST” as the teacher of the 80% student.  
**Do write:** Track 3 deliverable = KD-protect student 80%; AST ~90% is a separate research teacher; AST-KD student ~75%.

### Same stack, other verified runs (do not override table above)

| Run | Single (bvt) | Ensemble | Role |
|---|---:|---:|---|
| `server3090_notacher_f1_fullclip_baseline_50ep` | 76.90% | 75.40% | Same config; **weaker** than best single/ens — keep as server artifact, **not** research headline |
| AST-KD `local_fold1_kv260_ast_teacher_kd_50ep` | ~75.4% | ~77.6% | **Weaker** than Model B kdprotect; not the deliverable |
| Texture T1/T2/H2 | ≤74.6% bvt | up to 79.4% ens (T1) | **REJECTED** for single; T1 ens does not beat 79.89% best ens |

---

## What is in main

Git **`main`** carries:

1. **Code + MAIN config** above  
2. **This ACHIEVED table** (and REGISTRY pointers)  
3. **Not** every machine-specific folder — only the **declared best exp names** and metrics in docs / `results/*`  

Headline claims:

| Claim allowed now | Number |
|---|---|
| Best **single** no-teacher (research) | **79.08%** |
| Best **ensemble** no-teacher (research) | **79.89%** |
| Best **KD student** single (research) — Model B KD-protect | **80.00%** |
| Best **KD student** ensemble | **80.23%** |
| Teacher of Model B 80% run | same-family DS1D ~**79%** |
| AST teacher research (fold1 test, not deploy) | ~**90%** |
| AST train-cache fold2 (not deploy) | ~**92.4%** |

**Not** achieved: single **85%**, ensemble **85%**, KD student **85%**, multi-fold mean 80–85%.

---

## Gaps to close (still Phase A)

| Track | Gap to 80% | Gap to 85% |
|---|---:|---:|
| Single | **~0.9 pp** | **~5.9 pp** |
| Ensemble | **~0.1 pp** | **~5.1 pp** |
| KD student | **0** (at 80%) | **~5 pp** |

---

## File locations of best evidence

| Track | Path / branch |
|---|---|
| 1 & 2 | `experiments/local_multifold_pyramid_base_f1_f3_50ep/fold_1/metrics.json` |
| 3 | `experiments/local_finetune_kdprotect_f1_20ep/fold_1/metrics.json` |
| Server H0 (secondary) | `results/server3090-notacher-f1` → `.../server3090_notacher_f1_fullclip_baseline_50ep` |

When only one number is cited in thesis/main README, use **this file**, not “local vs server”.
