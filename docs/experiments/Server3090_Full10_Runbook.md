# Server 3090 Full-10 Runbook

> **Status:** optional/archived full-10 scheduling note. New Phase 1 dataset work starts from `main` and follows [`../data/MULTIDATASET_PHASE1.md`](../data/MULTIDATASET_PHASE1.md) plus [`../main/SERVER_POLICY.md`](../main/SERVER_POLICY.md).

Purpose:

```text
Run the three current best/diagnostic 1D-CNN candidates on all 10 source-safe
folds with 200 planned epochs, 4 cosine cycles, and validation early stopping.
```

Branch:

```bash
git fetch origin
git checkout main
git pull origin main
```

Check environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"
nvidia-smi
```

Expected GPU:

```text
NVIDIA RTX 3090
VRAM: about 24GB
```

The legacy 200-epoch server configs use:

```text
batch_size=128
amp=true
num_workers=0
```

This was intentionally conservative for avoiding dataloader/cache issues. For
new main/Phase 1 configs on `CPU-FPGA-GPU`, prefer the canonical server default
from [`../main/SERVER_POLICY.md`](../main/SERVER_POLICY.md): `batch_size=64`,
`num_workers=6`, `amp=true`. If CUDA OOM happens, rerun the same command with:

```bash
--batch_size 64 --skip_existing
```

## Automatic Screen Run

Use this option for the full unattended three-model run:

```bash
screen -S sound_full10
```

Inside screen:

```bash
conda activate sound_env
bash scripts/run_server3090_full10_three_models.sh
```

Detach from screen:

```text
Ctrl-a d
```

Re-attach:

```bash
screen -r sound_full10
```

If CUDA OOM happens, re-run with lower batch size. The script resumes completed
folds by default:

```bash
bash scripts/run_server3090_full10_three_models.sh --batch-size 64
```

Logs:

```text
logs/server3090_full10/
```

## Run 1: Current Best Pyramid Baseline

This run must finish before the protected fine-tune run because its checkpoints
are used as teacher and initialization checkpoints.

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_mixup_ema_200ep_es.json \
  --exp_name server3090_full10_pyramid_base_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes
```

Resume:

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_mixup_ema_200ep_es.json \
  --exp_name server3090_full10_pyramid_base_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes \
  --skip_existing
```

## Run 2: Lightweight AvgMax Mixup EMA

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_mixup_ema_200ep_es.json \
  --exp_name server3090_full10_avgmax_mixup_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes
```

Resume:

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_mixup_ema_200ep_es.json \
  --exp_name server3090_full10_avgmax_mixup_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes \
  --skip_existing
```

## Run 3: Protected Fine-Tune From Run 1

Only run this after Run 1 has written:

```text
experiments/server3090_full10_pyramid_base_200ep_es/fold_<N>/checkpoints/tcam_fold_<N>_cycle_final.pt
```

Command:

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_finetune_kdprotect_200ep_es.json \
  --exp_name server3090_full10_pyramid_finetune_kdprotect_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes
```

Resume:

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_finetune_kdprotect_200ep_es.json \
  --exp_name server3090_full10_pyramid_finetune_kdprotect_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes \
  --skip_existing
```

## Expected Outputs

Each run writes:

```text
experiments/<exp_name>/multifold_summary.md
experiments/<exp_name>/multifold_summary.json
experiments/<exp_name>/fold_<N>/metrics.json
experiments/<exp_name>/fold_<N>/analysis_all_cycles.json
```

The summary table includes:

```text
Epochs: completed/planned
Early stop: yes/no
Best validation accuracy
Validation-selected test accuracy
Final test accuracy
Last-2 ensemble test accuracy
Worst final class
```

## Early Stopping Policy

For the two scratch 200-epoch runs:

```text
warmup_epochs=80
patience=35
min_delta=0.001
monitor=val_clip_acc
```

For protected fine-tune:

```text
warmup_epochs=60
patience=30
min_delta=0.001
monitor=val_clip_acc
```

Early stopping is per fold. When it triggers, training still saves
`cycle_final.pt`, runs test evaluation, and writes `metrics.json`.

## Optional Research Hyperparameter Run

This is not part of the initial three-model full-10 batch. Run it after the
current batch produces enough evidence, or run fold 1 only as a diagnostic.

Research notes:

```text
docs/experiments/Hyperparameter_Research_Setup.md
```

Fold-1 diagnostic:

```bash
bash scripts/run_server3090_hparam_nesterov_step.sh
```

Full-10 promotion, only if fold 1 is better than the AdamW/cosine baseline:

```bash
bash scripts/run_server3090_hparam_nesterov_step.sh 1-10 server3090_full10_pyramid_nesterov_step_200ep_es
```
