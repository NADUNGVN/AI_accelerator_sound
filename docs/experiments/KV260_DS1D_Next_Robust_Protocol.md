# KV260 DS1D Next Robust Protocol

## Current Best Model

The current best model remains:

```text
Config: configs/kv260_ds1d_pyramid_mixup_ema_val.json
Model: KV260AudioNetDS1D
Width: 1.0
Pooling: pyramid_avgmax, bins [1, 2, 4]
Params: 101,674
MAC/clip: 61.85M
FLOPs-equivalent: 123.71M, using FLOPs = 2 * MACs
Validation-selected test accuracy: 79.08%
Final checkpoint test accuracy: 79.43%
Last-2 ensemble test accuracy: 79.89%
```

Final checkpoint per-class accuracy:

| Class | Accuracy |
|---|---:|
| air_conditioner | 65.00% |
| car_horn | 75.00% |
| children_playing | 70.00% |
| dog_bark | 68.00% |
| drilling | 90.00% |
| engine_idling | 85.00% |
| gun_shot | 100.00% |
| jackhammer | 90.82% |
| siren | 84.95% |
| street_music | 76.47% |

## Why The Next Step Is Multi-Fold First

The width `1.25` experiments showed that aggregate validation can miss a
class-specific regression. The control run reached the same best validation
accuracy as width `1.0`, but test accuracy and `jackhammer` were much worse:

```text
width 1.0 best val: 72.86%, final jackhammer: 90.82%
width 1.25 control best val: 72.86%, final jackhammer: 43.88%
```

Therefore, the next useful work is not another architecture change. It is to
measure stability across more source-group buckets.

## Research Basis

The next experiments follow these principles:

- UrbanSound8K is distributed with folds for reproducible comparison; a single
  split is not enough for a reliable final claim.
- Mixup trains on convex combinations of examples and labels, which is intended
  to improve generalization and reduce memorization.
- Between-Class learning was proposed specifically for deep sound recognition by
  mixing sounds from different classes and learning the mixing ratio.
- SpecAugment motivates simple masking policies for audio features; for the raw
  waveform model, the equivalent lightweight path is controlled time masking
  and waveform perturbation.

Sources:

```text
UrbanSound8K official dataset page:
https://urbansounddataset.weebly.com/urbansound8k.html

mixup:
https://arxiv.org/abs/1710.09412

Learning from Between-class Examples for Deep Sound Recognition:
https://arxiv.org/abs/1711.10282

SpecAugment:
https://arxiv.org/abs/1904.08779
```

## Experiment Order

### Phase 1: Multi-Fold Stability Of Current Best

Run folds `1-3` first. This is a cheap stability check before committing to all
10 folds.

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --exp_name multifold_pyramid_base_f1_f3_50ep --folds 1-3 --epochs 50 --analyze --eval_modes
```

If folds `1-3` look stable, run all 10 folds on the server:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --exp_name multifold_pyramid_base_f1_f10_50ep --folds 1-10 --epochs 50 --analyze --eval_modes
```

The summary files are written to:

```text
experiments/<exp_name>/multifold_summary.json
experiments/<exp_name>/multifold_summary.md
```

### Phase 2: General Robust Augmentation

Only run this if Phase 1 shows unstable class behavior.

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_general_robustaug_val.json --exp_name multifold_pyramid_general_robustaug_f1_f3_50ep --folds 1-3 --epochs 50 --analyze --eval_modes
```

This config keeps width `1.0` and the pyramid head, but removes manual
class-specific multipliers and uses slightly stronger general waveform
augmentation.

### Phase 3: Reject/Accept Criteria

Accept a candidate only if:

```text
mean validation-selected test accuracy improves by >=1 point
no critical class collapses by >15 points on any evaluated fold
params <=300K
MAC/clip <=300M
```

Critical classes:

```text
air_conditioner
children_playing
dog_bark
jackhammer
street_music
```

Reject a candidate immediately if it repeats the width `1.25` failure mode:

```text
aggregate validation looks acceptable
but one critical class collapses on test
```

## Current Decision

Do not continue width `1.25`, `1.50`, or `1.75` until multi-fold evidence says
the width `1.0` pyramid base is stable and the failure is not fold-specific.

The active base remains:

```text
configs/kv260_ds1d_pyramid_mixup_ema_val.json
```
