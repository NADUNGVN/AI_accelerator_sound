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

Source-level diagnostic command:

```bash
python tools/analyze_source_confusions.py --exp_name local_multifold_pyramid_base_f1_f3_50ep --folds 3 --classes air_conditioner,engine_idling,jackhammer,street_music --top_groups 8
```

Source-level finding on baseline fold 3:

| Class | fsID | Accuracy | Support | Main confusions |
|---|---|---:|---:|---|
| air_conditioner | 74677 | 0.00% | 31 | street_music 21, children_playing 7, jackhammer 3 |
| air_conditioner | 74507 | 0.00% | 25 | jackhammer 25 |
| air_conditioner | 202516 | 0.00% | 12 | engine_idling 12 |
| engine_idling | 177592 | 0.00% | 11 | siren 6, children_playing 2, street_music 2 |
| engine_idling | 111386 | 21.74% | 23 | car_horn 16, jackhammer 1, air_conditioner 1 |
| jackhammer | 132016 | 0.00% | 9 | children_playing 5, air_conditioner 4 |
| street_music | 93065 | 0.00% | 6 | children_playing 6 |
| street_music | 60608 | 0.00% | 6 | siren 6 |

This confirms that the accuracy drop is source-group concentrated. A candidate
that improves only average validation but still fails these source groups should
be rejected.

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

The third candidate tested was hard-negative margin training:

```text
Config: configs/kv260_ds1d_pyramid_hardneg_margin_val.json
Params: unchanged from baseline
MAC/clip: unchanged from baseline
Mechanism: add a small auxiliary loss during training only
Target: reduce confusion inside [air_conditioner, drilling, engine_idling, jackhammer, street_music]
```

Run command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_hardneg_margin_val.json --exp_name local_multifold_pyramid_hardneg_margin_f1_f3_50ep --folds 1-3 --epochs 50 --analyze --eval_modes
```

### Phase 3 Result: Hard-Negative Margin

Result files:

```text
experiments/local_multifold_pyramid_hardneg_margin_f1_f3_50ep/multifold_summary.json
experiments/local_multifold_pyramid_hardneg_margin_f1_f3_50ep/multifold_summary.md
experiments/local_multifold_pyramid_hardneg_margin_f1_f3_50ep/source_confusions.json
experiments/local_multifold_pyramid_hardneg_margin_f1_f3_50ep/source_confusions.md
```

Fold summary:

| Fold | Best val | Best-val test | Final test | Ensemble | Worst final class | Worst final acc |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 69.98% | 66.44% | 74.60% | 76.78% | air_conditioner | 45.00% |
| 2 | 69.23% | 70.44% | 70.44% | 71.02% | engine_idling | 38.00% |
| 3 | 71.38% | 67.74% | 67.85% | 68.08% | air_conditioner | 12.00% |

Aggregate versus baseline:

| Metric | Baseline mean | Hard-negative mean | Delta |
|---|---:|---:|---:|
| Validation-selected test | 71.23% | 68.20% | -3.02 |
| Final test | 71.88% | 70.96% | -0.91 |
| Last-2 ensemble | 72.30% | 71.96% | -0.34 |
| Worst final class | 42.00% | 31.67% | -10.33 |

Final test per-class delta:

| Class | Baseline mean | Hard-negative mean | Delta |
|---|---:|---:|---:|
| air_conditioner | 42.00% | 35.67% | -6.33 |
| car_horn | 86.54% | 84.89% | -1.65 |
| children_playing | 70.33% | 71.00% | +0.67 |
| dog_bark | 78.67% | 81.00% | +2.33 |
| drilling | 79.67% | 84.67% | +5.00 |
| engine_idling | 60.00% | 55.67% | -4.33 |
| gun_shot | 100.00% | 99.10% | -0.90 |
| jackhammer | 78.22% | 72.44% | -5.77 |
| siren | 83.03% | 85.20% | +2.17 |
| street_music | 67.77% | 67.10% | -0.67 |

Decision:

```text
Reject configs/kv260_ds1d_pyramid_hardneg_margin_val.json as a candidate.
It improved fold 2 but degraded fold 1 and fold 3. The main target class
air_conditioner dropped from 42.00% mean to 35.67% mean, and fold 3 dropped
from 18.00% to 12.00%.
```

Lesson:

```text
A pair/group margin on logits is too coarse for this problem. It pushes class
boundaries but does not learn the missing source-invariant representation for
stationary mechanical sounds.
```

### Phase 4 Candidate: Source-Invariant Supervised Contrastive Training

This candidate keeps the deployment model unchanged and adds a training-only
feature loss:

```text
Config: configs/kv260_ds1d_pyramid_supcon_sourceinv_val.json
Params: unchanged from baseline
MAC/clip: unchanged from baseline
Mechanism: CE + source-aware supervised contrastive feature loss
Sampler: batches contain same-class examples from different fsID source groups
Target: learn source-invariant features before the final classifier
```

Run command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_supcon_sourceinv_val.json --exp_name local_multifold_pyramid_supcon_sourceinv_f1_f3_50ep --folds 1-3 --epochs 50 --analyze --eval_modes
```

### Phase 4 Result: Source-Invariant SupCon

Result files:

```text
experiments/local_multifold_pyramid_supcon_sourceinv_f1_f3_50ep/multifold_summary.json
experiments/local_multifold_pyramid_supcon_sourceinv_f1_f3_50ep/multifold_summary.md
experiments/local_multifold_pyramid_supcon_sourceinv_f1_f3_50ep/source_confusions.json
experiments/local_multifold_pyramid_supcon_sourceinv_f1_f3_50ep/source_confusions.md
```

Fold summary:

| Fold | Best val | Best-val test | Final test | Ensemble | Worst final class | Worst final acc |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 72.63% | 78.28% | 78.97% | 79.31% | air_conditioner | 66.00% |
| 2 | 67.16% | 65.82% | 65.47% | 66.74% | jackhammer | 36.36% |
| 3 | 72.18% | 65.90% | 65.90% | 65.56% | air_conditioner | 27.00% |

Aggregate versus baseline:

| Metric | Baseline mean | SupCon mean | Delta |
|---|---:|---:|---:|
| Validation-selected test | 71.23% | 70.00% | -1.23 |
| Final test | 71.88% | 70.11% | -1.76 |
| Last-2 ensemble | 72.30% | 70.54% | -1.76 |
| Worst final class | 42.00% | 43.12% | +1.12 |

Final test per-class delta:

| Class | Baseline mean | SupCon mean | Delta |
|---|---:|---:|---:|
| air_conditioner | 42.00% | 43.33% | +1.33 |
| car_horn | 86.54% | 86.54% | +0.00 |
| children_playing | 70.33% | 68.33% | -2.00 |
| dog_bark | 78.67% | 78.67% | +0.00 |
| drilling | 79.67% | 77.33% | -2.33 |
| engine_idling | 60.00% | 59.00% | -1.00 |
| gun_shot | 100.00% | 100.00% | +0.00 |
| jackhammer | 78.22% | 61.05% | -17.16 |
| siren | 83.03% | 88.08% | +5.05 |
| street_music | 67.77% | 68.80% | +1.03 |

Important signal:

```text
SupCon improved the primary fold-3 air_conditioner collapse from 18.00% to
27.00%, and fsID 202516 improved from 0/12 to 9/12. However, fold 2 jackhammer
collapsed to 36.36%, causing aggregate accuracy to drop.
```

Decision:

```text
Reject configs/kv260_ds1d_pyramid_supcon_sourceinv_val.json as a candidate in
its current form. The idea has a useful signal for air_conditioner source
generalization, but the sampler/loss is too broad and damages jackhammer.
```

Next refinement:

```text
Do not use broad all-class SupCon with the current sampler. If this direction is
continued, restrict the objective to the air_conditioner/engine_idling boundary
or reduce SupCon weight, and protect jackhammer with a validation rejection
condition.
```

### Baseline 200-Epoch Fold-1 Check

Purpose:

```text
Check whether the 50-epoch baseline was simply under-trained compared with the
paper-style 200 epochs and 4 cycles.
```

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --exp_name local_pyramid_base_fold1_200ep --folds 1 --epochs 200 --analyze --eval_modes
```

Training schedule:

```text
epochs=200
cycles=4
snapshot_epochs=[50, 100, 150, 200]
```

Fold-1 result compared with the previous 50-epoch baseline:

| Run | Best val | Best-val test | Final test | Last-2 ensemble | Worst class |
|---|---:|---:|---:|---:|---|
| 50 epoch | 72.86% | 79.08% | 79.43% | 79.89% | air_conditioner 65.00% |
| 200 epoch | 68.71% | 76.21% | 76.55% | 76.67% | jackhammer 50.00% |

Cycle-level 200-epoch result:

| Snapshot | Test | Val | air_conditioner | engine_idling | jackhammer |
|---|---:|---:|---:|---:|---:|
| epoch 50 | 75.40% | 65.59% | 65.00% | 78.00% | 71.43% |
| epoch 100 | 75.63% | 66.63% | 67.00% | 89.00% | 53.06% |
| epoch 150 | 75.75% | 67.44% | 68.00% | 87.00% | 47.96% |
| epoch 200 | 76.55% | 66.17% | 66.00% | 87.00% | 50.00% |

Decision:

```text
Increasing the current baseline from 50 to 200 epochs does not improve fold 1.
The longer 4-cycle schedule lowers validation accuracy and damages jackhammer.
This supports the current diagnosis that the main bottleneck is source/class
generalization, not insufficient epoch count.
```

### Baseline 200-Epoch Fold-1 Paper-LR Check

Purpose:

```text
Check whether the 200-epoch failure above was caused by using the current
baseline LR=0.001 instead of the paper LR=0.0002.
```

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --exp_name local_pyramid_base_fold1_200ep_lr2e4 --folds 1 --epochs 200 --lr 0.0002 --analyze --eval_modes
```

Fold-1 comparison:

| Run | LR | Snapshot epochs | Best val | Best-val test | Final test | Last-2 ensemble | Worst class |
|---|---:|---|---:|---:|---:|---:|---|
| 50 epoch baseline | 0.001 | 13, 26, 39, 50 | 72.86% | 79.08% | 79.43% | 79.89% | air_conditioner 65.00% |
| 200 epoch baseline | 0.001 | 50, 100, 150, 200 | 68.71% | 76.21% | 76.55% | 76.67% | jackhammer 50.00% |
| 200 epoch paper-LR | 0.0002 | 50, 100, 150, 200 | 68.82% | 68.85% | 72.64% | 71.49% | air_conditioner 36.00% |

Cycle-level 200-epoch paper-LR result:

| Snapshot | Test | Val | air_conditioner | jackhammer | car_horn |
|---|---:|---:|---:|---:|---:|
| epoch 50 | 65.17% | 63.28% | 33.00% | 37.76% | 62.50% |
| epoch 100 | 68.62% | 65.70% | 35.00% | 50.00% | 60.00% |
| epoch 150 | 69.20% | 67.21% | 34.00% | 41.84% | 65.00% |
| epoch 200 | 72.64% | 67.90% | 36.00% | 52.04% | 65.00% |

Decision:

```text
Reject the paper-LR 200-epoch control for the current baseline. It is worse
than both the 50-epoch baseline and the 200-epoch lr=0.001 run on fold 1.
The 200-epoch paper-style schedule is therefore not the missing ingredient for
the current source-safe protocol.
```

### Source Group Audit

The next verification step is recorded in:

```text
docs/experiments/KV260_DS1D_Source_Group_Audit.md
```

Main conclusion:

```text
The current 72% multi-fold ceiling is dominated by source-domain failures.
For example, air_conditioner has 174 errors across folds 1-3, and its top
10 failed source groups account for 93.1% of those errors. Engine_idling top
10 source groups account for 89.2% of errors, and jackhammer top 5 source
groups account for 98.5% of errors.
```

Decision:

```text
Prioritize source-aware hard-group training and targeted weak-boundary
regularization before more long runs, width-only scaling, or broad SupCon.
```

### Local Source-Aware Training Result

Follow-up local runs are recorded in:

```text
docs/experiments/KV260_DS1D_Source_Group_Audit.md
```

Result:

```text
Source-hard CE was rejected after fold 1 because jackhammer collapsed from
90.82% to 60.20%. Source-balanced CE was then tested on folds 1-3 and did not
beat baseline: mean final 70.30% vs 71.88%, mean ensemble 69.46% vs 72.30%.
```

Decision update:

```text
Do not promote source-hard CE or source-balanced CE to 10 folds. Keep the
top-level source_aware_batch_sampler implementation for controlled ablations,
but the next improvement must address representation/source-domain features,
not only batch composition.
```
