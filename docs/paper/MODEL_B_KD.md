# Model B — KD-Student (canonical narrative)

**Status:** research truth for Track 3 deliverable (student **80.00%** / ensemble **80.23%**).  
**Do not conflate** with AST-KD student runs (~75% on fold 1).

---

## 1. What Model B is

| Item | Value |
|------|--------|
| Paper name | **DS-Conv2D-H1 Pyramid — KD-Student** |
| Topology | **Same** as Model A (deployable student) |
| Params / MACs | **101 674** / **61.85 M** per clip |
| Protocol | Source-Disjoint Protocol (**SDP 8-1-1**), seed **83**, fold **1** |
| Experiment | `local_finetune_kdprotect_f1_20ep` |
| Metric | `test_acc_best_val_model` = **80.00%**; ensemble last-2 = **80.23%** |
| Student bundle | `deploy/student_models/model_b_kd_student_80p00/` |
| Deploy | **Student only** — no teacher weights on chip / DPU |

---

## 2. Who is the teacher of the **80%** run?

**Not AST.** The headline KD run uses **same-family self-distillation (KD-protect)**.

| Role | Path / setting |
|------|----------------|
| Student init | `experiments/local_multifold_pyramid_base_f1_f3_50ep/fold_1/checkpoints/tcam_fold_1_cycle_final.pt` |
| **Teacher** | **Same** `tcam_fold_1_cycle_final.pt` (DS-Conv2D-H1 Pyramid) |
| Teacher config | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` |
| Recipe | Fine-tune **20** epochs, lr **2e-4**, KD weight **0.6**, T **2.0** |
| `protect_classes` | `[1, 4, 6, 7, 8]` — car_horn, drilling, gun_shot, jackhammer, siren |
| Config family | `configs/kv260_ds1d_pyramid_finetune_weakboost_kdprotect_val.json` (and related kdprotect configs) |

### Teacher strength (same-family)

From the No-Teacher lineage on the same fold (Model A exp):

| Metric | Value |
|--------|------:|
| Best-val test (`best.pt`) | **79.08%** |
| Ensemble | **79.89%** |
| Last snapshot (near `cycle_final`) | **~79.4%** |

After KD-protect fine-tune, the **student** reaches **80.00%** (ens **80.23%**).

```text
Model A train ──► cycle_final (~79%)
                      │
                      ├─ init student
                      └─ teacher soft labels  ──►  Model B student 80.00%
```

---

## 3. AST teacher (research track — separate)

AST was studied as a **large offline teacher** to probe accuracy ceiling under SDP. It is **not** the teacher wired into `local_finetune_kdprotect_f1_20ep`.

| Evidence | Number | Meaning |
|----------|-------:|---------|
| AST fine-tune fold 1 best-val test | **89.89%** | `local_ast_teacher_finetune_f1_12ep` |
| AST fine-tune fold 1 final test | **90.23%** | same run |
| AST fine-tune fold 1 best val | **89.26%** | checkpoint selection |
| AST train-cache fold 2 | **~92.44%** | accuracy on **cached train** logits (not deploy test) |
| AST embedding probe (f1–f3 mean, docs) | **~86%** | frozen embedding + linear/RBF |
| AST fine-tune fold 2 best-val test | **~85.3%** | harder fold |

**Base model:** HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` (AudioSet-pretrained AST).  
**Cost:** ~86M params, ~130G MAC — **not** KV260 student budget.  
**Deploy rule:** AST never ships; only student DS1D does.

### When AST was used to distill the **student**

| Run | Student best-val test | vs Model B |
|-----|----------------------:|------------|
| `local_fold1_kv260_ast_teacher_kd_50ep` | **~75.4%** | **Weaker** than 80% |
| Fold 2 AST-KD (docs) | **~72.2%** | Weaker |

→ AST raises the **teacher** ceiling; it has **not** produced the best **deployable student** on fold 1 in this repo.

---

## 4. Why AST was considered — and why Model B is not AST-KD

### Why study AST as teacher

1. **AudioSet pretraining** — strong environmental sound representation under source-safe splits.  
2. **Ceiling probe** — ~90% fold-1 test shows the task is not saturated at 79% student.  
3. **Literature-aligned KD** — large spectrogram Transformer → compact waveform CNN.  
4. **Clear deploy split** — heavy teacher offline; light student on board.

### Why the **deliverable** Model B uses same-family KD-protect instead

1. **Measured student accuracy** — 80.00% > AST-KD ~75% on fold 1.  
2. **Matched I/O** — teacher and student share waveform full-clip + same head; soft labels are “same language.”  
3. **Cheap reuse** — teacher is an existing No-Teacher snapshot; no AST fine-tune + logit cache required for this recipe.  
4. **Protect-classes KD** — stabilizes weak classes during fine-tune.  
5. **Track 3 definition** — success is **student** 80–85%, not teacher 90%+.

---

## 5. Claim rules (paper / thesis)

| Allowed | Not allowed |
|---------|-------------|
| “KD-Student (Model B) reaches **80.00%** best-val test under SDP fold 1 via **KD-protect** from a same-family teacher (~79%).” | “Model B is distilled from AST at 92%.” |
| “AST fine-tune reaches ~**90%** on fold 1 **as a teacher research model**.” | “Deploy accuracy is 90%+” (that is teacher, not student). |
| “AST-KD student fold 1 ~**75%** (secondary, weaker).” | Equating train-cache **92.4%** with test deploy accuracy. |

---

## 6. Pointers

| Resource | Path |
|----------|------|
| Student metrics | `experiments/local_finetune_kdprotect_f1_20ep/fold_1/metrics.json` |
| Teacher snapshot (DS1D) | `experiments/local_multifold_pyramid_base_f1_f3_50ep/fold_1/checkpoints/tcam_fold_1_cycle_final.pt` |
| AST teacher summary | `experiments/local_ast_teacher_finetune_f1_12ep/fold_1/summary.md` |
| AST-KD notes | `docs/experiments/AST_Teacher_Distillation_KV260.md` |
| Achieved table | `docs/main/ACHIEVED.md` |
| Chip bundle | `deploy/student_models/model_b_kd_student_80p00/` |
