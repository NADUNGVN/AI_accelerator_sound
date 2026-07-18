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

## Local Execution Results

Machine:

```text
Windows local machine
Python: C:\Users\Dawin\AppData\Local\Programs\Python\Python311\python.exe
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

### Phase 1 Result: Current Best Baseline

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --exp_name local_multifold_pyramid_base_f1_f3_50ep --folds 1-3 --epochs 50 --analyze --eval_modes
```

Result files:

```text
experiments/local_multifold_pyramid_base_f1_f3_50ep/multifold_summary.json
experiments/local_multifold_pyramid_base_f1_f3_50ep/multifold_summary.md
```

Fold summary:

| Fold | Best val | Best-val test | Final test | Ensemble | Worst final class | Worst final acc |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 72.86% | 79.08% | 79.43% | 79.89% | air_conditioner | 65.00% |
| 2 | 70.49% | 67.67% | 67.32% | 68.82% | air_conditioner | 43.00% |
| 3 | 69.43% | 66.93% | 68.89% | 68.20% | air_conditioner | 18.00% |

Aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| Best validation | 70.93% | 1.44% |
| Validation-selected test | 71.23% | 5.56% |
| Final test | 71.88% | 5.38% |
| Last-2 ensemble | 72.30% | 5.37% |
| Worst final class | 42.00% | 19.20% |

Final test per-class mean:

| Class | Mean | Std |
|---|---:|---:|
| air_conditioner | 42.00% | 19.20% |
| car_horn | 86.54% | 9.17% |
| children_playing | 70.33% | 10.21% |
| dog_bark | 78.67% | 8.58% |
| drilling | 79.67% | 7.85% |
| engine_idling | 60.00% | 18.06% |
| gun_shot | 100.00% | 0.00% |
| jackhammer | 78.22% | 11.18% |
| siren | 83.03% | 12.94% |
| street_music | 67.77% | 7.06% |

Main finding:

```text
The single-fold best result is not stable. Fold 2 and fold 3 drop to the high
60s. The dominant failure is air_conditioner under source changes, especially
fold 3 where air_conditioner reaches only 18/100.
```

Weak-class confusions:

| Experiment | Fold | Weak class | Accuracy | Main confusions |
|---|---:|---|---:|---|
| baseline | 1 | air_conditioner | 65.00% | children_playing 19, engine_idling 6, car_horn 5 |
| baseline | 2 | air_conditioner | 43.00% | drilling 22, engine_idling 15, siren 11 |
| baseline | 2 | engine_idling | 43.00% | drilling 14, children_playing 12, street_music 12 |
| baseline | 3 | air_conditioner | 18.00% | jackhammer 29, engine_idling 25, street_music 21 |
| baseline | 3 | engine_idling | 52.00% | car_horn 17, jackhammer 8, air_conditioner 7 |

Interpretation:

```text
The model is still learning source- and texture-specific cues. The weak region is
not model size; it is the boundary among stationary/mechanical/background-rich
classes: air_conditioner, engine_idling, jackhammer, drilling, and street_music.
```

### Phase 2 Result: General Robust Augmentation

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_general_robustaug_val.json --exp_name local_multifold_pyramid_general_robustaug_f1_f3_50ep --folds 1-3 --epochs 50 --analyze --eval_modes
```

Result files:

```text
experiments/local_multifold_pyramid_general_robustaug_f1_f3_50ep/multifold_summary.json
experiments/local_multifold_pyramid_general_robustaug_f1_f3_50ep/multifold_summary.md
```

Fold summary:

| Fold | Best val | Best-val test | Final test | Ensemble | Worst final class | Worst final acc |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 71.02% | 76.67% | 76.44% | 76.09% | jackhammer | 58.16% |
| 2 | 73.02% | 68.13% | 68.24% | 68.82% | engine_idling | 38.00% |
| 3 | 67.59% | 69.46% | 69.46% | 68.77% | air_conditioner | 25.00% |

Aggregate:

| Metric | Baseline mean | Robust mean | Delta |
|---|---:|---:|---:|
| Validation-selected test | 71.23% | 71.42% | +0.19 |
| Final test | 71.88% | 71.38% | -0.50 |
| Last-2 ensemble | 72.30% | 71.23% | -1.07 |
| Worst final class | 42.00% | 40.39% | -1.61 |

Final test per-class delta:

| Class | Baseline mean | Robust mean | Delta |
|---|---:|---:|---:|
| air_conditioner | 42.00% | 43.33% | +1.33 |
| car_horn | 86.54% | 84.08% | -2.46 |
| children_playing | 70.33% | 73.00% | +2.67 |
| dog_bark | 78.67% | 80.33% | +1.67 |
| drilling | 79.67% | 82.67% | +3.00 |
| engine_idling | 60.00% | 58.33% | -1.67 |
| gun_shot | 100.00% | 100.00% | +0.00 |
| jackhammer | 78.22% | 64.98% | -13.23 |
| siren | 83.03% | 84.83% | +1.81 |
| street_music | 67.77% | 68.73% | +0.97 |

Decision:

```text
Reject configs/kv260_ds1d_pyramid_general_robustaug_val.json as a candidate.
It does not improve mean final accuracy, does not improve worst-class accuracy,
and causes a large jackhammer regression.
```

## Current Decision

Do not continue width `1.25`, `1.50`, or `1.75`.

Do not continue the first general robust-augmentation config.

The active base remains:

```text
configs/kv260_ds1d_pyramid_mixup_ema_val.json
```

The next useful direction is data/representation targeted, not simple size
increase:

1. diagnose source buckets for stationary/mechanical classes;
2. add a training-time objective that improves separation among
   `air_conditioner`, `engine_idling`, `jackhammer`, `drilling`, and
   `street_music`;
3. keep the deployment budget at `params <=300K` and `MAC/clip <=300M`;
4. evaluate every candidate with at least folds `1-3` before spending server
   time on all 10 folds.
