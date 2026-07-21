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

## Width 1.25 Control Verification

Run:

```bash
python train.py --fold 1 --config configs/kv260_ds1d_pyramid_w125_control_val.json --exp_name local_pyramid_w125_control_50ep --epochs 50
python tools/analyze_experiment.py --exp_dir experiments/local_pyramid_w125_control_50ep/fold_1 --fold 1 --config configs/kv260_ds1d_pyramid_w125_control_val.json --eval_all_cycles --eval_modes
```

Profile:

```text
Params: 148,930
MAC/clip: 90.52M
FLOPs-equivalent: 181.04M
Budget guard: pass
```

Overall result:

| Run | Params | MAC/clip | Best val | Val-selected test | Final test | Ensemble |
|---|---:|---:|---:|---:|---:|---:|
| width 1.0 pyramid | 101,674 | 61.85M | 72.86% | 79.08% | 79.43% | 79.89% |
| width 1.25 weakmixup | 148,930 | 90.52M | 70.79% | 73.33% | 75.29% | 75.06% |
| width 1.25 control | 148,930 | 90.52M | 72.86% | 74.14% | 75.06% | 74.94% |

Jackhammer result:

| Run | Final test acc | Jackhammer acc | Main jackhammer confusion |
|---|---:|---:|---|
| width 1.0 pyramid | 79.43% | 90.82% | drilling 3, air_conditioner 2, car_horn 2 |
| width 1.25 weakmixup | 75.29% | 41.84% | air_conditioner 40, gun_shot 5, street_music 5 |
| width 1.25 control | 75.06% | 43.88% | air_conditioner 41, dog_bark 6, gun_shot 3 |

Conclusion:

- `w125_control` does **not** restore `jackhammer`.
- Restoring the original mixup, dropout, and weight decay improves validation
  versus `w125_weakmixup`, but it does not fix the class-specific test
  regression.
- The regression is now more likely caused by the interaction between width
  `1.25`, EMA validation, and source-specific mechanical texture boundaries.
- Aggregate validation accuracy is not sufficient here: `w125_control` reaches
  the same best validation accuracy as width `1.0` (`72.86%`) but has much worse
  test accuracy and jackhammer accuracy.

Updated decision:

- Do not run width `1.50` or `1.75` until `jackhammer` is protected.
- The next repair test should be `w125_jackguard`.
- If `w125_jackguard` still fails, return to width `1.0` pyramid and optimize
  class-pair augmentation rather than widening.

## Width 1.25 Jackguard Verification

Run:

```bash
python train.py --fold 1 --config configs/kv260_ds1d_pyramid_w125_jackguard_val.json --exp_name local_pyramid_w125_jackguard_50ep --epochs 50
python tools/analyze_experiment.py --exp_dir experiments/local_pyramid_w125_jackguard_50ep/fold_1 --fold 1 --config configs/kv260_ds1d_pyramid_w125_jackguard_val.json --eval_all_cycles --eval_modes
```

Profile:

```text
Params: 148,930
MAC/clip: 90.52M
FLOPs-equivalent: 181.04M
Budget guard: pass
```

Overall result:

| Run | Params | MAC/clip | Best val | Val-selected test | Final test | Ensemble |
|---|---:|---:|---:|---:|---:|---:|
| width 1.0 pyramid | 101,674 | 61.85M | 72.86% | 79.08% | 79.43% | 79.89% |
| width 1.25 weakmixup | 148,930 | 90.52M | 70.79% | 73.33% | 75.29% | 75.06% |
| width 1.25 control | 148,930 | 90.52M | 72.86% | 74.14% | 75.06% | 74.94% |
| width 1.25 jackguard | 148,930 | 90.52M | 72.86% | 69.54% | 70.00% | 70.46% |

Final checkpoint mechanical-texture comparison:

| Run | Test acc | air_conditioner | engine_idling | jackhammer | Main jackhammer confusion |
|---|---:|---:|---:|---:|---|
| width 1.0 pyramid | 79.43% | 65.00% | 85.00% | 90.82% | drilling 3, air_conditioner 2, car_horn 2 |
| width 1.25 weakmixup | 75.29% | 65.00% | 85.00% | 41.84% | air_conditioner 40, gun_shot 5, street_music 5 |
| width 1.25 control | 75.06% | 66.00% | 71.00% | 43.88% | air_conditioner 41, dog_bark 6, gun_shot 3 |
| width 1.25 jackguard | 70.00% | 32.00% | 78.00% | 46.94% | air_conditioner 43, car_horn 3, gun_shot 2 |

Cycle-level note for `w125_jackguard`:

| Checkpoint | Test acc | Val acc | air_conditioner | jackhammer |
|---|---:|---:|---:|---:|
| cycle 1 | 46.90% | 55.54% | 40.00% | 23.47% |
| cycle 2 | 70.92% | 69.28% | 33.00% | 69.39% |
| cycle 3 | 70.92% | 71.82% | 32.00% | 57.14% |
| final | 70.00% | 72.06% | 32.00% | 46.94% |

Conclusion:

- `w125_jackguard` does not solve the issue.
- It can temporarily raise test `jackhammer` to `69.39%` at cycle 2, but this
  comes with a severe collapse of `air_conditioner` to `33.00%`.
- Final `jackhammer` remains only `46.94%`, far below the width `1.0` pyramid
  result of `90.82%`.
- The added jackhammer weight shifts the mechanical-texture boundary rather than
  learning a robust distinction.

Final decision for width `1.25`:

- Stop the current width `1.25` expansion path.
- Do not run `w150` or `w175` with this architecture/training recipe.
- Revert the active base to `configs/kv260_ds1d_pyramid_mixup_ema_val.json`.

Next solution direction:

1. **Use width `1.0` pyramid as the stable base.**
   - It is still the best validated model: `79.08%` validation-selected test and
     `79.89%` ensemble.

2. **Improve class boundaries through data/augmentation, not width.**
   - Target class pairs:
     - `air_conditioner` vs `engine_idling`;
     - `jackhammer` vs `air_conditioner`;
     - `children_playing` vs `street_music`;
     - `dog_bark` vs `siren/children_playing`.

3. **Add class-pair diagnostics before another architecture change.**
   - Track per-class validation and test deltas for every run.
   - Reject a run if any critical class drops more than 15 points even when
     aggregate validation looks acceptable.

4. **Use multi-fold source-group evaluation before claiming progress.**
   - A single validation bucket missed the `jackhammer` test collapse.
   - Rotate validation/test buckets or run multiple folds before selecting a
     final architecture.
