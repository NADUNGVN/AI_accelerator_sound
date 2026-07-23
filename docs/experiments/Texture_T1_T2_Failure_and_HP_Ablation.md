# Texture T1/T2 failure analysis and controlled HP ablation

**Status:** historical/rejected texture analysis; use git `main` for any rerun.
**Locked model:** `kv260_audio_net_ds1d` full-clip  
**Full write-up (workspace guides):** see local `TEXTURE_T1_T2_FAILURE_ANALYSIS_AND_HP_PLAN.md` if present.

## Why selection failed

| Run | Best-val test | Ensemble | Root issue |
|---|---:|---:|---|
| Baseline | **76.90%** | 75.40% | Control |
| T1 sourcebalance | 73.56% | **79.43%** | Source-aware changes train dist → val is weaker selection proxy |
| T2 sourcehard | 74.60% | 77.01% | Multi-change confound + industrial hard-neg pairs collapse jackhammer |

T2 changed together: source-aware multipliers, hard-neg (14 directed pairs including jackhammer), class CE weights, mixup 0.7→0.4, stronger augment. Not attributable.

T2 top confusion included **jackhammer → air_conditioner (28)** after pairs `(0,7),(7,0),(4,7),(7,4)`.

## Controlled next configs

| ID | Config | Single intent |
|---|---|---|
| H2 | `configs/kv260_ds1d_pyramid_hneg_ac_engine_val.json` | Hard-neg **only** AC↔engine on pure baseline |
| H3 | `configs/kv260_ds1d_pyramid_sourcebalance_hneg_ac_engine_val.json` | T1 + same narrow hard-neg |
| H4 | `configs/kv260_ds1d_pyramid_sourcehard_v2_safe_val.json` | Source-aware uniform + HN without industrial pairs; mixup/aug = baseline |

### Run order on 3090

1. H2 → 2. H3 → 3. H4 (optional)

```bash
git fetch origin
git checkout main
git pull origin main
nohup python tools/run_multifold.py \
  --config configs/kv260_ds1d_pyramid_hneg_ac_engine_val.json \
  --exp_name server3090_texture_h2_hneg_ac_engine_f1_50ep \
  --folds 1 --epochs 50 --analyze --eval_modes \
  > logs/texture_hp/h2.nohup.log 2>&1 &
```

### Pre-registered gate (fold 1)

- Primary: `test_acc_best_val_model` vs baseline **76.90%**
- Safety: worst-class ≥ ~63%; do not destroy jackhammer/drilling
- Secondary: AC→engine confusion count vs baseline 19

No teacher/KD. No architecture change in this loop.
