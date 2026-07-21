# Tight research protocol: texture / source-safe (no-teacher)

**Locked:** `kv260_audio_net_ds1d` full-clip, ~101.7k params, no teacher/KD  
**Data protocol:** `source_group_8_1_1` + `fsid_classid_balanced_v1`, seed **83**, fold **1** first  
**Primary metric (pre-registered):** `test_acc_best_val_model` (best-val → test)  
**Baseline B:** **76.90%** best-val test (`results/server3090-notacher-f1`)  
**Secondary:** ensemble, worst-class acc, count AC→engine on final-cycle confusion  
**Safety floor:** worst-class ≥ **63%** (do not accept jackhammer collapse like T2)

---

## A. Dataset-grounded problem (why we intervene)

| Dataset fact | Implication |
|---|---|
| Many clips share `fsID` | Model can memorize recording texture |
| Source-safe split → 0 train/test source overlap | Random-split accuracy is not the target |
| Fold1 confusions: AC↔engine, street↔children, industrial family | Intervene on **those** pairs only |
| Full-clip 1×64k | Silent-tail is not the main lever here |

**Do not** change architecture or add teacher in this protocol.

---

## B. What already failed (evidence)

| Run | Change vs H0 | Best-val test | Lesson |
|---|---|---:|---|
| H0 baseline | control | **76.90%** | selection works |
| T1 | source-aware only | 73.56% | train≠val proxy; ensemble↑ still selection-fail |
| T2 | many HPs + wide hard-neg | 74.60% | **confounded** + industrial pairs → jackhammer 61% |

**Rule:** one causal axis per run after T2.

---

## C. Phase plan (strict order)

### Phase 1 — Causal ablation (lr frozen = 1e-3)

| Step | Config | Exp name | Hypothesis |
|---|---|---|---|
| **P1.1 H2** | `configs/kv260_ds1d_pyramid_hneg_ac_engine_val.json` | `server3090_texture_h2_hneg_ac_engine_f1_50ep` | Narrow HN AC↔engine only helps confusion without breaking selection |
| **P1.2 H3** | `configs/kv260_ds1d_pyramid_sourcebalance_hneg_ac_engine_val.json` | `server3090_texture_h3_srcbal_hneg_ac_engine_f1_50ep` | Only if H2 safety OK: add source-aware on top of H2 HN |
| **P1.3 H4** | `configs/kv260_ds1d_pyramid_sourcehard_v2_safe_val.json` | `server3090_texture_h4_sourcehard_v2_safe_f1_50ep` | Optional cleanup of T2 idea; **skip if H2/H3 already reject** |

**H2 only changes:** hard-neg pairs `[[0,5],[5,0]]`, weight `0.02`, margin `0.30`.  
Everything else = baseline (lr 1e-3, mixup 0.7, aug, CE weights).

### Phase 2 — LR sensitivity (only if Phase 1 needs it)

**Gate to enter Phase 2** (any one):

- H2 best-val test ∈ `[B−1.5, B)` **and** train looks under-optimized (low train dynamics), or  
- H2 best-val test < `B−1.5` but confusions improve and loss/grad scale suspect, or  
- H3 selection-fail like T1 (ensemble↑, best-val↓) → try lower lr on **best Phase-1 config only**

| Config | lr | Exp name |
|---|---:|---|
| `configs/kv260_ds1d_pyramid_hneg_ac_engine_lr5e-4_val.json` | 5e-4 | `server3090_texture_h2_lr5e-4_f1_50ep` |
| `configs/kv260_ds1d_pyramid_hneg_ac_engine_lr2e-3_val.json` | 2e-3 | `server3090_texture_h2_lr2e-3_f1_50ep` |

**Only one axis:** lr. Same HN as H2.  
Do **not** LR-sweep T2 soup. Do **not** run Phase 2 before H2 finishes.

### Phase 3 — Stop / scale

| Outcome | Action |
|---|---|
| Any config ≥ B and worst ≥ 63% | Candidate; then folds 1–3 **same config only** |
| All Phase 1–2 < B−1.0 | Freeze interventions; write negative result; keep H0 as deploy baseline |
| Partial (confusion↓, selection flat) | At most **one** re-tune (weight 0.01 or margin 0.2), not a new zoo |

---

## D. Fixed training constants (unless Phase 2)

```text
optimizer AdamW | lr 0.001 | batch 64 | wd 0.0008
epochs 50 | seed 83 | cosine_restart (as trainer default for this family)
mixup alpha 0.2 prob 0.7 | EMA 0.995
no teacher | no architecture change
```

**Honesty note:** lr=1e-3 is **control inheritance**, not re-derived from dataset in Phase 1. Phase 2 exists exactly because of that.

---

## E. Analysis checklist after each run

1. `best_val_clip_acc`, epoch of best val  
2. `test_acc_best_val_model` vs **76.90%**  
3. `test_acc_ensemble`, last snapshot  
4. worst-class name + acc  
5. Top confusions: AC→engine count vs baseline **19**  
6. Confirm `source_label_overlap_train_test.count == 0`  
7. Confirm `distillation is null`

Decision uses **(2)+(4)** primarily; (5) explains mechanism.

---

## F. Git / server hygiene

- Code branch: `research/fpga-1dcnn-90acc`  
- Results: force-add JSON/MD only on `results/server3090-texture-f1` (or new results branch)  
- Never `git add .` for experiments (ignored); never commit `*.pt`
