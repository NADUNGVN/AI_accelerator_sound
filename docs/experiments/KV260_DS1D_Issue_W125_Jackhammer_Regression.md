# Issue: Width 1.25 Jackhammer Regression

## Summary

The first expanded-budget model passed the KV260 compute budget but reduced
accuracy. The failure is class-specific: `jackhammer` collapsed on the test set.

Run:

```text
Config: configs/kv260_ds1d_pyramid_w125_weakmixup_val.json
Experiment: local_pyramid_w125_weakmixup_50ep
Protocol: source_group_8_1_1
Params: 148,930
MAC/clip: 90.52M
FLOPs-equivalent: 181.04M
Budget: pass under 300K params and 300M MAC/clip
```

Result:

```text
Best validation accuracy: 70.79%
Validation-selected test accuracy: 73.33%
Final checkpoint test accuracy: 75.29%
Last-2 ensemble test accuracy: 75.06%
```

Previous best:

```text
Config: configs/kv260_ds1d_pyramid_mixup_ema_val.json
Params: 101,674
MAC/clip: 61.85M
Validation-selected test accuracy: 79.08%
Final checkpoint test accuracy: 79.43%
```

## Symptom

Width `1.25` improved some broad classes but severely hurt `jackhammer`.

Final checkpoint comparison:

| Class | Width 1.0 pyramid | Width 1.25 weakmixup | Change |
|---|---:|---:|---:|
| air_conditioner | 65.00% | 65.00% | +0.00 |
| children_playing | 70.00% | 79.00% | +9.00 |
| dog_bark | 68.00% | 71.00% | +3.00 |
| jackhammer | 90.82% | 41.84% | -48.98 |
| street_music | 76.47% | 78.43% | +1.96 |

The key confusion:

```text
Width 1.25 final test jackhammer:
support=98, correct=41, accuracy=41.84%
main confusion: air_conditioner=40
```

The model did not fail uniformly. It shifted the decision boundary for a
mechanical texture class while still improving some human/background mixture
classes.

## Evidence From Training Dynamics

Width `1.0` pyramid:

| Epoch | Train acc | Val acc |
|---:|---:|---:|
| 13 | 48.95% | 46.65% |
| 26 | 56.95% | 71.13% |
| 39 | 57.74% | 72.63% |
| 50 | 64.65% | 72.86% |

Width `1.25` weakmixup:

| Epoch | Train acc | Val acc |
|---:|---:|---:|
| 13 | 57.90% | 55.77% |
| 26 | 64.12% | 69.98% |
| 39 | 72.58% | 69.05% |
| 50 | 72.75% | 69.17% |

Interpretation:

- Width `1.25` learned the training split faster and reached higher training
  accuracy.
- Validation did not improve and eventually decreased.
- This pattern is consistent with weaker source-independent generalization, not
  a compute-budget problem.

## Likely Causes

### 1. The experiment changed more than width

The width `1.25` run was not a pure width ablation.

Changed at the same time:

| Setting | Width 1.0 best | Width 1.25 weakmixup |
|---|---:|---:|
| width_mult | 1.0 | 1.25 |
| dropout | 0.25 | 0.28 |
| weight_decay | 0.0008 | 0.001 |
| mixup alpha | 0.20 | 0.10 |
| mixup prob | 0.70 | 0.50 |

Because mixup was weakened while capacity increased, the model had more ability
to fit source-specific texture cues. The result cannot be attributed to width
alone.

### 2. Jackhammer is source-group sensitive

`jackhammer` has few test source groups in this split. A model can look
reasonable on validation source groups but fail on held-out test source groups.

Width `1.25` final:

```text
val jackhammer: 67.68%
test jackhammer: 41.84%
```

The gap shows that validation jackhammer and test jackhammer are not identical
distribution samples. This class needs robustness across sources, not only
greater capacity.

### 3. The wider model shifted mechanical texture boundaries

The main test error is `jackhammer -> air_conditioner`.

This suggests the wider model learned smoother stationary texture cues that
overlap with `air_conditioner`, instead of preserving the intermittent impact
structure that separated `jackhammer` in the width `1.0` pyramid model.

### 4. Validation-selected checkpoint does not protect this failure mode enough

The best validation checkpoint has:

```text
validation-selected test accuracy: 73.33%
final checkpoint test accuracy: 75.29%
```

Validation still helps avoid direct test tuning, but this run shows that a
single validation bucket can miss class-specific test regressions. For final
claims, the model should be evaluated across multiple folds or validation
buckets.

## Resolution Plan

Do not continue to width `1.50` or `1.75` with the same weakmixup recipe.

Run controlled repairs in this order:

1. **Width-isolation control**
   - Keep width `1.25`.
   - Restore the original width `1.0` training recipe:
     - mixup `alpha=0.20`, `prob=0.70`;
     - weight decay `0.0008`;
     - dropout `0.25`.
   - Purpose: determine whether width itself hurts, or the weaker regularization
     caused the regression.

2. **Jackhammer guard**
   - Keep width `1.25`.
   - Use stronger mixup.
   - Increase only the `jackhammer` class weight moderately.
   - Purpose: recover mechanical texture separation without returning to the
     aggressive sampler recipe that previously hurt `street_music` and
     `drilling`.

3. **If both fail**
   - Return to width `1.0` pyramid as the base.
   - Focus on class-pair augmentation for:
     - `air_conditioner` vs `engine_idling`;
     - `children_playing` vs `street_music`;
     - `dog_bark` vs `siren/children_playing`.

4. **Before final thesis claim**
   - Run more than one source-group fold or rotate validation/test buckets.
   - A single fold can overstate or understate class-specific behavior.

## New Candidate Configs

```text
configs/kv260_ds1d_pyramid_w125_control_val.json
configs/kv260_ds1d_pyramid_w125_jackguard_val.json
```

Both keep the expanded budget guard:

```text
max_params: 300,000
max_macs_per_clip: 300,000,000
```

Smoke verification:

```text
configs/kv260_ds1d_pyramid_w125_control_val.json
  params: 148,930
  MAC/clip: 90.52M
  budget guard: pass

configs/kv260_ds1d_pyramid_w125_jackguard_val.json
  params: 148,930
  MAC/clip: 90.52M
  budget guard: pass
```

Next full run priority:

1. Run `w125_control` first. This isolates whether width `1.25` itself causes
   the regression.
2. Run `w125_jackguard` only if `w125_control` still loses `jackhammer`.
