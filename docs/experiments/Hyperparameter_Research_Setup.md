# Hyperparameter Research Setup

Purpose:

```text
Verify the current 1D-CNN training setup against related environmental-sound
papers, then add one controlled hyperparameter run that can be compared against
the current AdamW/cosine baseline without changing the model budget.
```

## Research Basis

The current thesis target is stricter than reproducing a paper:

```text
1D-CNN only
UrbanSound8K first
source-safe split, no clip/source leakage
accuracy target >90%
params <=300K
MAC/clip target <=300M
Kria KV260 deployment direction
```

Primary sources checked:

| Source | Relevant setup | Implication for this repo |
|---|---|---|
| Xu et al., 2024, TCAM1DCNN, Expert Systems With Applications, DOI: https://doi.org/10.1016/j.eswa.2024.123768 | Adam, LR 0.0002, batch 100, MSLE, 200 epochs, 4 cosine cycles, last-two snapshot ensemble. Reported UrbanSound8K 91.43% single / 94.04% ensemble. | This exact direction was already tested under source-safe splitting and did not explain the gap. Keep it as reproduction evidence, not the next expensive run. |
| Abdoli et al., 2019, raw-waveform 1D-CNN: https://arxiv.org/abs/1904.08990 | UrbanSound8K audio downsampled to 16 kHz; one training fold used for validation; batch 100; up to 100 epochs with early stopping; Adadelta LR 1.0; reported 89% mean accuracy. | Supports batch 100 and validation/early stopping. Also shows 89% is possible for raw 1D-CNN without the 2024 TCAM schedule, but still below the thesis >90% target. |
| Salamon & Bello, 2017: https://arxiv.org/abs/1608.04363 | Cross-entropy with mini-batch SGD, batch 100, constant LR 0.01, dropout 0.5, L2 0.001, 50 epochs, validation fold selection. Augmentations help, but class impact differs. | Supports testing SGD-style optimization instead of only Adam/AdamW. Also warns that augmentation should not be blindly strengthened for every weak class. |
| Tokozume et al., 2018, BC learning: https://arxiv.org/abs/1711.10282 | Original folds; random crop/pad; input-space mixing; Nesterov momentum 0.9, weight decay 0.0005, batch 64. UrbanSound8K learning settings commonly use step LR drops by 10x. | Supports Nesterov + weight_decay 0.0005 + multistep LR as a meaningful optimizer hypothesis. It also supports mixup/BC-style input mixing, but only if trained long enough. |
| Zhang et al., 2018, mixup: https://arxiv.org/abs/1710.09412 | Trains on convex combinations of examples and labels; alpha controls mixing strength. | Current `alpha=0.20` is reasonable. Do not increase it before measuring optimizer/schedule effects. |
| Loshchilov & Hutter, AdamW: https://arxiv.org/abs/1711.05101 | Decoupled weight decay is a principled Adam variant. | Current AdamW baseline is not arbitrary, but related ESC papers give enough reason to run one Nesterov control. |

## Current Setup Verification

The currently running server config remains valid as the first full-10 baseline:

```text
config: configs/server3090_kv260_ds1d_pyramid_mixup_ema_200ep_es.json
optimizer: AdamW
lr: 0.001
schedule: cosine restart, 4 cycles
batch_size: 128
epochs: 200 planned
early stopping: warmup 80, patience 35, min_delta 0.001
params: about 101.7K
MAC/clip: about 61.9M
```

Do not interrupt that run only because it is not identical to a paper. It is the
best local source-safe 1D-CNN family so far, and it is already producing the
10-fold evidence needed to know whether the 50-epoch fold-1 baseline was lucky
or representative.

## Added Research Config

New config:

```text
configs/server3090_kv260_ds1d_pyramid_nesterov_step_200ep_es.json
```

This is a controlled optimizer/schedule test:

```text
model: same KV260 DS1D pyramid model
input: full 4-second raw waveform at 16 kHz, one 64,000-sample frame per clip
optimizer: Nesterov SGD
momentum: 0.9
weight_decay: 0.0005
batch_size: 100
lr: 0.01
lr_schedule: multistep, drop by 10x after epochs 100 and 150
loss: crossentropy
label_smoothing: 0.0
mixup: alpha 0.20, prob 0.70
EMA: enabled, decay 0.995
early stopping: warmup 60, patience 35, min_delta 0.001
```

Why this is the next useful test:

```text
The paper-style Adam LR=0.0002/MSLE path has already failed under source-safe
conditions. The untested paper-backed axis is optimizer dynamics: Nesterov SGD
with weight decay and step LR. It changes the training behavior while keeping
architecture, params, MAC/clip, data split, and augmentation largely fixed.
```

Reject condition:

```text
If fold 1 cannot approach the AdamW baseline validation/test range by epoch
60-100, do not promote it to a full 10-fold run.
```

Promotion condition:

```text
Promote only if fold 1 improves validation-selected test accuracy or repairs
weak classes without collapsing jackhammer/air_conditioner.
```

## Commands

Smoke test:

```bash
python train.py --fold 1 \
  --config configs/server3090_kv260_ds1d_pyramid_nesterov_step_200ep_es.json \
  --exp_name smoke_nesterov_step \
  --epochs 1 \
  --max_train_clips 40 \
  --max_val_clips 10 \
  --max_test_clips 10
```

Single-fold diagnostic:

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_nesterov_step_200ep_es.json \
  --exp_name server3090_pyramid_nesterov_step_fold1_200ep_es \
  --folds 1 \
  --epochs 200 \
  --analyze \
  --eval_modes \
  --skip_existing
```

Full-10 run, only after the diagnostic is worth promoting:

```bash
python tools/run_multifold.py \
  --config configs/server3090_kv260_ds1d_pyramid_nesterov_step_200ep_es.json \
  --exp_name server3090_full10_pyramid_nesterov_step_200ep_es \
  --folds 1-10 \
  --epochs 200 \
  --analyze \
  --eval_modes \
  --skip_existing
```

