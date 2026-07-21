# Config index

Configs stay in `configs/*.json` (stable paths for scripts/history).  
**Use status below** before launching a run.

| Status | Meaning |
|---|---|
| **MAIN** | Default for deploy/thesis |
| **REJECTED** | Failed vs H0 or confounded — do not use as main |
| **OPTIONAL** | Side experiments / KD / literature |
| **LEGACY** | Older reproduce/TCAM paths |

---

## MAIN

| File | Protocol | Role |
|---|---|---|
| **`kv260_ds1d_pyramid_mixup_ema_val.json`** | `source_group_8_1_1` | **Canonical** full-clip DS1D + mixup + EMA + **val** + best-val selection |

---

## REJECTED (texture / hard-neg line)

| File | Why |
|---|---|
| `kv260_ds1d_pyramid_sourcebalance_ce_val.json` | T1 server lost best-val test |
| `kv260_ds1d_pyramid_sourcehard_ce_val.json` | T2 confounded + jackhammer |
| `kv260_ds1d_pyramid_hneg_ac_engine_val.json` | H2 lost best-val test |
| `kv260_ds1d_pyramid_hneg_ac_engine_lr5e-4_val.json` | H2-gated LR; H2 rejected |
| `kv260_ds1d_pyramid_hneg_ac_engine_lr2e-3_val.json` | same |
| `kv260_ds1d_pyramid_sourcebalance_hneg_ac_engine_val.json` | H3 — do not run after H2 fail |
| `kv260_ds1d_pyramid_sourcehard_v2_safe_val.json` | H4 — do not run after H2 fail |
| `kv260_ds1d_pyramid_hardneg_margin_val.json` | Prior hard-neg line |
| `kv260_ds1d_pyramid_supcon_sourceinv_val.json` | SupCon source-inv rejected multifold |

---

## OPTIONAL — official literature

| File | Protocol | Role |
|---|---|---|
| `kv260_ds1d_pyramid_mixup_ema_paper91.json` | `paper_9_1` | Optional side table; **not** deploy main |

---

## OPTIONAL — KD / teacher (later)

| File | Role |
|---|---|
| `kv260_ds1d_pyramid_ast_teacher_kd_val.json` | AST-KD student |
| `kv260_ds1d_pyramid_ast_teacher_kd_smoke.json` | Smoke only |
| `kv260_ds1d_pyramid_finetune_*kdprotect*.json` | Finetune + KD protect family |
| `kv260_ds1d_pyramid_weakboost_kdprotect_val.json` | KD variant |
| `accuracy_first_logmel_teacher_*.json` | Logmel teacher family |
| `server3090_kv260_ds1d_pyramid_finetune_kdprotect_200ep_es.json` | Long server KD |

---

## OPTIONAL / LEGACY — architecture & reproduce

| File | Role |
|---|---|
| `kv260_ds1d_pyramid_frame16k_*.json` | Frame ablations (lost to full-clip) |
| `kv260_ds1d_mil_attention_no_teacher_f1.json` | MIL (not main) |
| `kv260_ds1d_pyramid_w125_*.json` | Width sweeps |
| `kv260_ds1d_deep_pyramid_*.json` / `lateres2_*` | Depth/late residual |
| `kv260_dsafe_*.json` | DSafe variants |
| `reproduce_msle.json` / `reproduce_crossentropy.json` | Paper reproduce TCAM-style |
| `proposed_*.json` | Early proposed configs |
| `source_group_msle.json` / `random_clip_msle.json` | Split controls |
| `server3090_kv260_ds1d_pyramid_mixup_ema_200ep_es.json` | Long H0-like server |
| `rtx3090_config.json` | Hardware-oriented legacy |

---

## Rule

New experiment → either **MAIN config** or a **new file** with a new name + register in `docs/experiments/REGISTRY.md`.  
Do not silently edit MAIN without updating `docs/main/DECISIONS_LOG.md`.
