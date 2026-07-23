# Experiment registry

**Purpose:** single index of runs so we **do not repeat failed recipes**.  
Artifacts live under `experiments/<exp_name>/` (gitignored locally; some metrics on `results/*` branches).

**Status legend**

| Status | Meaning |
|---|---|
| **BASELINE** | Canonical numbers for main path |
| **REJECTED** | Do not re-run as main improvement |
| **OPTIONAL** | Side path / literature / later KD |
| **DIAG** | Smoke/debug only |
| **ACTIVE** | Allowed next work |

---

## BASELINE / BEST ACHIEVED (main — one main truth)

Do **not** headline “local vs server”. Use [../main/ACHIEVED.md](../main/ACHIEVED.md).

| Role | exp_name | Single (bvt) | Ensemble | Config |
|---|---|---:|---:|---|
| **Best single + best ens (no-teacher)** | `local_multifold_pyramid_base_f1_f3_50ep` fold1 | **79.08%** | **79.89%** | `kv260_ds1d_pyramid_mixup_ema_val.json` |
| Duplicate of above numbers | `local_priority_pyramid_mixup_ema_50ep` fold1 | 79.08% | 79.89% | same |
| **Best KD student** | `local_finetune_kdprotect_f1_20ep` fold1 | **80.00%** | **80.23%** | kdprotect family |
| Weaker same-stack artifact | `server3090_notacher_f1_fullclip_baseline_50ep` | 76.90% | 75.40% | same no-teacher config — **not** headline |

Multi-fold note: base f1–3 ens mean ~**72.3%** (not yet 80–85% mean).

---

## REJECTED (do not re-run as main)

| exp_name | Branch / place | Config | Result vs H0 | Lesson |
|---|---|---|---|---|
| `server3090_texture_sourcebalance_f1_50ep` | `results/server3090-texture-f1` | `..._sourcebalance_ce_val.json` | bvt **73.56%** | Source-aware alone hurts selection; ens can rise |
| `server3090_texture_sourcehard_f1_50ep` | same | `..._sourcehard_ce_val.json` | bvt **74.60%**; jackhammer **61%** | Multi-HP + industrial hard-neg confounded |
| `server3090_texture_h2_hneg_ac_engine_f1_50ep` | same | `..._hneg_ac_engine_val.json` | bvt **74.14%**; AC→engine worse | Narrow HN does not beat H0 |
| `local_sourcehard_ce_f1_50ep` | local | sourcehard | under baseline story | Aligns with server T2 lesson |
| `local_multifold_pyramid_supcon_sourceinv_f1_f3_50ep` | local | supcon | f1–3 ens ~70.5% | Broad SupCon rejected |
| `local_multifold_pyramid_general_robustaug_f1_f3_50ep` | local | robustaug | f1–3 ens ~71.2% | Not over base mean |
| `local_mil_attention_no_teacher_f1_30ep` | local | MIL | ens ~71.5%, **491k** params | Not main deploy model |
| `local_frame16k_sum_f1_50ep` / `..._duration...` | local | frame16k | below full-clip | Full-clip remains main |

**Do not start:** H3/H4 texture stack, H2 LR sweeps, re-run T1/T2 without new hypothesis.

---

## DIAG / optional official

| exp_name | Branch | Status | Notes |
|---|---|---|---|
| `server3090_official_paper91_ds1d_fold1_50ep` | `results/server3090-official-paper91-smoke` | **DIAG** | paper_9_1, last **~66.8%**, ens **~67.1%**; **not** main; do not full-10 by default |
| config `kv260_ds1d_pyramid_mixup_ema_paper91.json` | research branch | **OPTIONAL** | Literature side-table only |

---

## OPTIONAL (KD / teacher) — later, not “new backbone”

| exp_name | Status | Notes |
|---|---|---|
| `local_ast_teacher_finetune_f1_12ep` (+ f2) | OPTIONAL | Teacher ckpts exist |
| `ast_teacher_logits_sourcegroup_train/fold_{1,2}` | OPTIONAL | Reuse logits; teacher train ~92% (f2 doc) |
| `local_fold1_kv260_ast_teacher_kd_50ep` | OPTIONAL | Student f1 bvt ~75.4% |
| `local_fold2_kv260_ast_teacher_kd_50ep` | OPTIONAL | Student f2 bvt ~72.2% |
| `local_finetune_kdprotect_f1_20ep` | OPTIONAL | Student f1 bvt **80.0%**, ens **80.2%** — best KD signal |
| `local_finetune_kdprotect_f1_f3_20ep` | OPTIONAL | Mean f1–3 ~72% — fold1 alone not enough |

**Rule:** reuse teacher/logits if present; do not retrain AST without reason.

---

## ACTIVE naming for new runs

```text
server3090_<topic>_<protocolTag>_f{folds}_<ep>ep
```

Examples:

- `server3090_deploy_h0_refresh_f1_50ep` — only if intentional re-baseline  
- `server3090_kd_kdprotect_f1_50ep` — KD track  

**Forbidden names:** reuse `server3090_texture_*` recipes above without new hypothesis id.

---

## Git results branches (metrics only)

| Branch | Content |
|---|---|
| `results/server3090-notacher-f1` | H0 baseline |
| `results/server3090-texture-f1` | T1, T2, H2 |
| `results/server3090-official-paper91-smoke` | paper91 fold1 diag |

Push: `git add -f` metrics/history/predictions/analysis/summary — **never** `*.pt` via `git add .` alone.
