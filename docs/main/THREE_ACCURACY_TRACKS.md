# Three accuracy tracks (primary goals)

**Before** SoC design, quantization, and KV260 deploy, hit these targets.

**One research truth** (no local vs server split in claims):  
→ [ACHIEVED.md](ACHIEVED.md)

**Shared stack (main)**

| Item | Lock |
|---|---|
| Backbone | **DS-Conv2D-H1 Pyramid** (`ds_conv2d_h1_pyramid`; legacy `kv260_audio_net_ds1d`) full-clip (**101 674** params) |
| Config (no-teacher) | `configs/main/student_ds_conv2d_h1_pyramid_sourcegroup.json` |
| Protocol | `source_group_8_1_1`, seed **83**, val + best-val selection |
| Single metric | `test_acc_best_val_model` |
| Ensemble metric | `test_acc_ensemble` (last-2) |
| Target band | **80–85%** |

---

## Track 1 — Single model **80–85%**

| | |
|---|---|
| Meaning | One checkpoint (best val); one forward |
| Metric | best-val → test |
| **Best achieved** | **79.08%** |
| Gap to 80% / 85% | **~0.9 pp** / **~5.9 pp** |

## Track 2 — Ensemble **80–85%**

| | |
|---|---|
| Meaning | Last-2 snapshots (same model family) |
| Metric | ensemble test |
| **Best achieved** | **79.89%** |
| Gap to 80% / 85% | **~0.1 pp** / **~5.1 pp** |

## Track 3 — Distill teacher → 1D-CNN **80–85%**

| | |
|---|---|
| Meaning | Teacher ~90%+ → KD → student DS1D deployable |
| Metric | **student** best-val test |
| **Best achieved** | Student **80.00%** (ens **80.23%**); teacher ~**90%+** |
| Gap to 85% student | **~5 pp** |

---

## Order

```text
Phase A:  Track1 single → Track2 ensemble → Track3 KD student   (all 80–85%)
Phase B:  SoC → quantization → KV260 deploy
```

Tracks 1–2 share runs; Track 3 is KD recipe.  
Deploy = **one student ckpt**, not teacher weights.

## Not primary until Phase A done

- Official paper_9_1 full-10 as the main goal  
- REJECTED texture recipes as the path to 80–85%  
- Teacher 90%+ as “model accuracy” on device  
