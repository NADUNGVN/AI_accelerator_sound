# Three accuracy tracks (primary goals)

**Before** SoC design, quantization, and KV260 deploy, hit these accuracy targets on the locked protocol.

**Shared baseline stack**

| Item | Lock |
|---|---|
| Student / deploy backbone | `kv260_audio_net_ds1d` full-clip (~101.7k) |
| Config family | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` (+ KD configs when Track 3) |
| Protocol | `source_group_8_1_1`, seed 83, **val + best-val selection** |
| Primary single-model metric | `test_acc_best_val_model` |
| Ensemble metric | `test_acc_ensemble` (last-2 cycle snapshots) |
| Target band | **80–85%** (fold1 first; then multi-fold when stable) |

---

## Track 1 — Single model (one checkpoint) **80–85%**

| | |
|---|---|
| Meaning | **One** weight file after **best-val** selection; one forward at test |
| Metric | `test_acc_best_val_model` |
| Current evidence | Local ~**79.1%**; server H0 ~**76.9%** → gap **~1–3 pp** to 80%, larger to 85% |
| Out of scope for this track | Reporting ensemble as if it were single; random-split inflation |

## Track 2 — Ensemble **80–85%**

| | |
|---|---|
| Name | **Ensemble** (last-2 / multi-snapshot; not a second architecture) |
| Meaning | Average/vote **≥2** cycle checkpoints (same DS1D family) |
| Metric | `test_acc_ensemble` |
| Current evidence | Local full-clip ~**79.9%** → **already near 80%**; server H0 ens ~**75.4%** |
| Deploy note | Optional research score; board may still ship **one** distilled/selected ckpt |

## Track 3 — Distill teacher → 1D-CNN student **80–85%**

| | |
|---|---|
| Meaning | Teacher (e.g. AST, train/cache ~**90%+**) → soft labels / KD → **student DS1D** still ~102k |
| Metric | Student **best-val test** (and ens secondary); teacher acc is **not** deploy acc |
| Current evidence | Teacher AST train/cache ~**92%** (f2 doc); student kdprotect f1 ~**80.0–80.2%**; AST-KD f1/f2 lower on some folds |
| Rule | Reuse teacher ckpt/logits when present; val/test **without** teacher at inference |

---

## Order

```text
1) Track 1 — single 80–85%
2) Track 2 — ensemble 80–85%   (can measure on same runs as Track 1)
3) Track 3 — KD student 80–85%
4) SoC / quantization / KV260 deploy   ← only after accuracy tracks are credible
```

Tracks 1 and 2 often share one training run (report both metrics).  
Track 3 is a separate training recipe (distillation on).

## Explicitly not primary (until tracks land)

- Official `paper_9_1` full-10 as the only goal  
- Re-running REJECTED texture T1/T2/H2 as the path to 80–85%  
- Claiming teacher 90%+ as the deployed model accuracy  

## Registry

Update [../experiments/REGISTRY.md](../experiments/REGISTRY.md) with track tag: `T1-single` / `T2-ens` / `T3-kd` on new exps.
