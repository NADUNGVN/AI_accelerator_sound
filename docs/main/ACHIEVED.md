# Achieved (research truth — no local vs server split)

**Rule:** One research stack. Report the **best verified run** for each goal.  
Machine (laptop / 3090) only matters for *where files live*, not for *which number is official research*.

**Main path stack (in git `research` → to be `main`)**

| Item | Value |
|---|---|
| Model | `kv260_audio_net_ds1d` full-clip |
| Params / MAC | **101 674** / **61 854 400** |
| Config (no-teacher) | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` |
| Protocol | `source_group_8_1_1`, seed **83**, train/val/test, best-val checkpoint |
| Primary single metric | `test_acc_best_val_model` |
| Ensemble metric | `test_acc_ensemble` (last-2) |

---

## Three tracks — target vs **best achieved**

| Track | Target | **Best achieved** | Exp (canonical evidence) | Status vs 80–85% |
|---|---|---:|---|---|
| **1. Single model** | 80–85% | **79.08%** best-val test | `local_multifold_pyramid_base_f1_f3_50ep` / fold_1 (same recipe as H0 config) | **Below 80%** (~−0.9 pp) |
| **2. Ensemble** | 80–85% | **79.89%** ensemble | same exp fold_1 | **Near 80%** (~−0.1 pp) |
| **3. KD student** | 80–85% | **80.00%** best-val test / **80.23%** ens | `local_finetune_kdprotect_f1_20ep` / fold_1 | **Hits 80% band** (not yet 85%) |

**Teacher (not deploy acc):** AST train/cache evidence ~**92.4%** (fold2 logits doc) — used only for Track 3 training.

### Same stack, other verified runs (do not override table above)

| Run | Single (bvt) | Ensemble | Role |
|---|---:|---:|---|
| `server3090_notacher_f1_fullclip_baseline_50ep` | 76.90% | 75.40% | Same config; **weaker** than best single/ens — keep as server artifact, **not** research headline |
| AST-KD `local_fold1_kv260_ast_teacher_kd_50ep` | ~75.4% | ~77.6% | Weaker than kdprotect on fold1 |
| Texture T1/T2/H2 | ≤74.6% bvt | up to 79.4% ens (T1) | **REJECTED** for single; T1 ens does not beat 79.89% best ens |

---

## What is “in main” once promoted

Git **`main`** (after promote) must carry:

1. **Code + MAIN config** above  
2. **This ACHIEVED table** (and REGISTRY pointers)  
3. **Not** every machine-specific folder — only the **declared best exp names** and metrics in docs / `results/*`  

Headline claims:

| Claim allowed now | Number |
|---|---|
| Best **single** no-teacher (research) | **79.08%** |
| Best **ensemble** no-teacher (research) | **79.89%** |
| Best **KD student** single (research) | **80.00%** |
| Best **KD student** ensemble | **80.23%** |
| Teacher (train/cache, not deploy) | **~90%+** |

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
