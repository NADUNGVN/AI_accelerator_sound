# Main setup (deploy / thesis) — re-locked

**Date:** 2026-07-21  
**Branch:** `research/fpga-1dcnn-90acc`

## Dropped from the main path

| Item | Status |
|---|---|
| Official `paper_9_1` full-10 as **primary** goal | **Out of main path** |
| Train 9 folds / test 1 / **no val** as deploy protocol | **Out** |
| Using last-snapshot-only as “best model” for hardware | **Out** (not primary) |
| Config `configs/kv260_ds1d_pyramid_mixup_ema_paper91.json` | **Optional / archive only** — not blocking deploy |
| Smoke `server3090_official_paper91_ds1d_fold1_50ep` (~67% last) | **Diagnostic only** — do not scale to full-10 for main story |

Rationale: paper target is **hardware deployment** (KV260 / 1D-CNN student).  
Official 10-fold without val is a **literature leaderboard** protocol, not required to select or flash one deploy model.

## Main path (active)

| Item | Choice |
|---|---|
| **Model** | `kv260_audio_net_ds1d` full-clip |
| **Params / MAC** | ~101.7k / ~61.9M |
| **Config** | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` |
| **Protocol** | `source_group_8_1_1` + `fsid_classid_balanced_v1` |
| **Seed** | `83` (reproducibility of splits/train; keep) |
| **Data split** | Train / **Val** / Test (8+1+1 source buckets) |
| **Checkpoint rule** | **Best validation accuracy** → report **test** of that checkpoint |
| **Secondary metric** | Last snapshot, last-2 ensemble (optional tables) |
| **Teacher** | Optional later (KD); deploy = student only |
| **Texture T1/T2/H2** | Rejected vs H0 — do not re-run as main |

### Primary numbers to report (main path)

| Source | Best-val → test | Ensemble (secondary) |
|---|---:|---:|
| Local fold1 (existing) | ~79.08% | ~79.89% |
| Server H0 fold1 (existing) | **~76.90%** | ~75.40% |

### Hardware story (main)

- One checkpoint after best-val selection  
- Params / MAC within budget  
- Source-safe = closer to “new recording / new device” than random split  

## What “seed” still means

Keep **`seed: 83`** in config: controls split + init reproducibility.  
That is **not** the dropped “paper_9_1 seed of work”.

## Next runs (main path only)

1. **No** full-10 `paper_9_1` unless a reviewer later demands a side table.  
2. Continue from **source-safe H0** as baseline (already on server).  
3. Next experiments: only if they improve **best-val test** under same protocol (or KD student later).  
4. Push results with `git add -f` on metrics JSON/MD only.

## Server quick check

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
git pull origin research/fpga-1dcnn-90acc
test -f configs/kv260_ds1d_pyramid_mixup_ema_val.json && echo OK_MAIN
# protocol must be source_group_8_1_1 in that file
```
