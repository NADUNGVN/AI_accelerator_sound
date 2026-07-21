# KV260 DS1D Train-Time Protection And Frame Check

Date: 2026-07-19

## Goal

Test train-time optimization paths that keep inference suitable for KV260:

```text
Model family: 1D-CNN
Parameter budget: <=300K
MAC/clip budget: <=300M
Target accuracy: >90%, but only under a source-safe split with no fsID+classID leakage
```

Current reference baseline:

```text
Config: configs/kv260_ds1d_pyramid_mixup_ema_val.json
Params: 101,674
MAC/clip: 61.85M
FLOPs convention: 2 * MACs = 123.71M FLOPs/clip
Folds 1-3 final mean: 71.88%
Folds 1-3 ensemble mean: 72.30%
Fold 1 final / ensemble: 79.43% / 79.89%
```

## Code Changes

Implemented two training-only controls:

```text
distillation:
  Load a fold-specific teacher checkpoint and add protected KL distillation.
  The teacher is used only during training, so inference params/MACs are unchanged.

initial_checkpoint:
  Initialize the student from a fold-specific checkpoint before training.
  This is used for protected fine-tuning from the current baseline.
```

Touched files:

```text
src/training/trainer.py
train.py
```

New configs:

```text
configs/kv260_ds1d_pyramid_weakboost_kdprotect_val.json
configs/kv260_ds1d_pyramid_finetune_weakboost_kdprotect_val.json
configs/kv260_ds1d_pyramid_frame16k_sum_val.json
configs/kv260_ds1d_pyramid_frame16k_duration_sum_val.json
```

## Experiment 1: Scratch Weak-Boost Protected KD

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_weakboost_kdprotect_val.json --exp_name local_kdprotect_weakboost_f1_50ep --folds 1 --epochs 50 --analyze --eval_modes
```

Result:

| Metric | Baseline fold 1 | KD weak-boost fold 1 |
|---|---:|---:|
| Final test | 79.43% | 77.82% |
| Last-2 ensemble | 79.89% | 77.13% |
| air_conditioner | 65.00% | 87.00% |
| drilling | 90.00% | 93.00% |
| engine_idling | 85.00% | 84.00% |
| jackhammer | 90.82% | 50.00% |

Decision:

```text
Reject.
```

Finding:

```text
The weak-class reweighting improved air_conditioner, but it moved the
air_conditioner/jackhammer boundary too aggressively. Logit KD from a random
student was not enough to preserve the baseline representation.
```

## Experiment 2: Baseline-Initialized Protected Fine-Tune

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_finetune_weakboost_kdprotect_val.json --exp_name local_finetune_kdprotect_f1_f3_20ep --folds 1-3 --epochs 20 --analyze --eval_modes
```

Result:

| Metric | Baseline mean | Fine-tune KD mean | Delta |
|---|---:|---:|---:|
| Best validation | 70.93% | 71.54% | +0.61 |
| Validation-selected test | 71.23% | 72.18% | +0.95 |
| Final test | 71.88% | 72.22% | +0.34 |
| Last-2 ensemble | 72.30% | 72.22% | -0.08 |
| Worst final class | 42.00% | 43.00% | +1.00 |

Per-class final mean:

| Class | Baseline | Fine-tune KD | Delta |
|---|---:|---:|---:|
| air_conditioner | 42.00% | 43.00% | +1.00 |
| car_horn | 86.54% | 83.18% | -3.36 |
| children_playing | 70.33% | 71.00% | +0.67 |
| dog_bark | 78.67% | 79.00% | +0.33 |
| drilling | 79.67% | 81.33% | +1.66 |
| engine_idling | 60.00% | 61.67% | +1.67 |
| gun_shot | 100.00% | 100.00% | +0.00 |
| jackhammer | 78.22% | 78.21% | -0.01 |
| siren | 83.03% | 81.95% | -1.08 |
| street_music | 67.77% | 67.78% | +0.01 |

Decision:

```text
Do not promote yet.
```

Finding:

```text
Initializing from the baseline and using stronger KD prevents jackhammer
collapse, but the mean improvement is too small. This is useful as a protection
mechanism for future train-time optimization, not as the main accuracy path.
```

## Experiment 3: Paper-Style 16K Frames With SUM

Hypothesis:

```text
The paper reports strong results using multiple short waveform frames and SUM
aggregation. Re-test that idea with the KV260 DS1D backbone and source-safe
splits.
```

Config:

```text
configs/kv260_ds1d_pyramid_frame16k_sum_val.json
frame_length=16000
frame_hop=4000
frames_per_clip=13
```

Model profile:

```text
Params: 101,674
MAC/frame: 15.51M
MAC/clip: 201.57M
FLOPs convention: 2 * MACs = 403.13M FLOPs/clip
```

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_frame16k_sum_val.json --exp_name local_frame16k_sum_f1_50ep --folds 1 --epochs 50 --analyze --eval_modes
```

Result:

| Metric | Baseline fold 1 | Frame16K fold 1 |
|---|---:|---:|
| Final test | 79.43% | 70.69% |
| Last-2 ensemble | 79.89% | 71.95% |
| Worst class | air_conditioner 65.00% | jackhammer 31.63% |

Decision:

```text
Reject.
```

Finding:

```text
Naive frame labeling is harmful. Short events such as gun_shot and car_horn
produce many padded or non-event frames, but every frame inherits the clip
label during training.
```

## Experiment 4: Duration-Aware 16K Frames

Duration audit for 16K/4K/13 frames:

| Class | Avg duration | Avg valid frames |
|---|---:|---:|
| air_conditioner | 3.99s | 12.97 |
| car_horn | 2.46s | 7.34 |
| dog_bark | 3.15s | 9.67 |
| gun_shot | 1.65s | 3.59 |
| street_music | 4.00s | 12.98 |

Config:

```text
configs/kv260_ds1d_pyramid_frame16k_duration_sum_val.json
drop_silent_tail_frames=true
eval_drop_silent_tail_frames=true
```

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_frame16k_duration_sum_val.json --exp_name local_frame16k_duration_f1_30ep --folds 1 --epochs 30 --analyze --eval_modes
```

Result:

| Metric | Baseline fold 1 | Duration-aware frame16K fold 1 |
|---|---:|---:|
| Final test | 79.43% | 77.24% |
| Last-2 ensemble | 79.89% | 77.13% |
| air_conditioner | 65.00% | 69.00% |
| children_playing | 70.00% | 87.00% |
| siren | 84.95% | 93.55% |
| jackhammer | 90.82% | 63.27% |

Decision:

```text
Reject as a replacement for the full-clip baseline.
```

Finding:

```text
Duration-aware frame training fixes part of the label-noise problem and improves
some localized or high-energy classes. It still damages jackhammer heavily and
does not beat the full-clip baseline on fold 1.
```

## Current Decision

The best deployable model remains:

```text
configs/kv260_ds1d_pyramid_mixup_ema_val.json
Params: 101,674
MAC/clip: 61.85M
FLOPs convention: 123.71M FLOPs/clip
Folds 1-3 final mean: 71.88%
Folds 1-3 ensemble mean: 72.30%
```

Protected fine-tuning remains useful as a guardrail for future experiments, but
it does not solve the >90% target.

## Next Research Direction

Do not spend more time on:

```text
random-init weak boost + KD
16K frame SUM as a replacement for full-clip input
duration-aware 16K frames as a replacement for full-clip input
```

Next useful work:

```text
1. Keep the full-clip DS1D baseline as the main branch.
2. Add train-time protection only when testing risky changes.
3. Investigate a full-clip representation improvement, not pure frame voting:
   - front-end filterbank/Sinc-style temporal kernels,
   - auxiliary frame consistency without replacing full-clip inference,
   - or a stronger but still <=300K DS1D full-clip model with guarded fine-tune.
4. Add QAT only after a candidate improves FP32 accuracy; QAT should protect
   deployment accuracy, not be expected to fix source-domain accuracy by itself.
```
