# Official paper_9_1 full-10: DS1D pyramid mixup+EMA

> **Status (2026-07-21): NOT the main path.**  
> Deploy/thesis primary setup is **source-safe + val + best-val checkpoint**:  
> `configs/kv260_ds1d_pyramid_mixup_ema_val.json` and  
> `docs/experiments/MAIN_SETUP_DEPLOY_SOURCE_SAFE.md`.  
> This paper_9_1 runbook is **optional literature side-table only**. Do not block hardware work on full-10.

## Goal

Journal-comparable UrbanSound8K evaluation (optional):

- Protocol: **`paper_9_1`** (train 9 official folds, test 1 fold, **no validation fold**)
- Model: **`kv260_audio_net_ds1d`** (~101.7k params)
- Config: `configs/kv260_ds1d_pyramid_mixup_ema_paper91.json`
- Checkpoint rule: **last snapshot** (primary), **last-2 ensemble** (secondary)
- Folds: **1–10**, report **mean ± std**

This is separate from `source_group_8_1_1` research numbers (best-val test).

## Metrics to report

| Field in metrics.json | Role |
|---|---|
| `test_acc_last_snapshot` | **Primary** single-model test (final snapshot) |
| `test_acc_ensemble` | Secondary (last-2 cycles) |
| `test_acc_best_val_model` | N/A / ignore under paper_9_1 |
| `uses_validation` | Must be `false` |
| `protocol` | Must be `paper_9_1` |

Multifold summary maps:

- `final_test_acc_pct` ← last snapshot  
- `ensemble_test_acc_pct` ← ensemble  

## Server commands (CPU-FPGA-GPU)

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
hostname   # CPU-FPGA-GPU
conda activate sound_env
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

git checkout main
git pull origin main

test -f configs/kv260_ds1d_pyramid_mixup_ema_paper91.json && echo OK_CONFIG
test -d data/UrbanSound8K || tar -xzf data/UrbanSound8K_on_server.tar.gz -C data

mkdir -p logs/official_paper91
# Prefer screen: screen -S paper91

python tools/run_multifold.py \
  --config configs/kv260_ds1d_pyramid_mixup_ema_paper91.json \
  --exp_name server3090_official_paper91_ds1d_full10_50ep \
  --folds 1-10 \
  --epochs 50 \
  --analyze \
  --eval_modes \
  2>&1 | tee logs/official_paper91/full10.log
```

Optional: smoke fold 1 only first:

```bash
python tools/run_multifold.py \
  --config configs/kv260_ds1d_pyramid_mixup_ema_paper91.json \
  --exp_name server3090_official_paper91_ds1d_fold1_smoke_50ep \
  --folds 1 \
  --epochs 50 \
  --analyze \
  --eval_modes
```

## After train — push results

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
EXP=server3090_official_paper91_ds1d_full10_50ep
BRANCH=results/server3090-official-paper91-full10
MSG="Add server3090 official paper_9_1 DS1D full10 results"

git fetch origin
git checkout -B $BRANCH origin/main

# Force-add metrics only (experiments/ is gitignored; do NOT add .pt)
for f in 1 2 3 4 5 6 7 8 9 10; do
  git add -f \
    experiments/$EXP/fold_$f/metrics.json \
    experiments/$EXP/fold_$f/history.json \
    experiments/$EXP/fold_$f/predictions.json \
    experiments/$EXP/fold_$f/analysis_all_cycles.json 2>/dev/null || true
done
git add -f experiments/$EXP/multifold_summary.json experiments/$EXP/multifold_summary.md

git status
git commit -m "$MSG"
git push -u origin $BRANCH
```

Then notify for analysis: report mean **Final test** (= last snapshot) and **Ensemble** from `multifold_summary.md`.
