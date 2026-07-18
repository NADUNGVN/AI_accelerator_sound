# KV260 DS1D Priority Approach Sweep

## Technical Summary

This sweep tested five non-width improvement directions for the deployable
Conv2D-H1 1D-CNN family on UrbanSound8K with source-group validation. The goal
was to improve accuracy without leaving the KV260-oriented budget.

The best result is:

```text
Config: configs/kv260_ds1d_pyramid_mixup_ema_val.json
Model: KV260AudioNetDS1D
Params: 101,674
MAC/clip: 61.85M
Validation-selected test accuracy: 79.08%
Final checkpoint test accuracy: 79.43%
Last-2 snapshot ensemble test accuracy: 79.89%
```

The clean validation-selected improvement over the previous baseline is:

```text
76.21% -> 79.08% = +2.87 points
```

This is still not close to the final `>90%` target. The useful discovery is
that temporal aggregation is the strongest lever found so far: keeping
coarse temporal regions in the classifier head improves source-independent
generalization more than focal loss or an early multi-scale stem.

## Scope And Metric Definitions

Task:

```text
Dataset: UrbanSound8K
Classes: 10 standard classes
Input: mono raw waveform, 16 kHz, 4 seconds, 64,000 samples
Model family: logical 1D-CNN implemented as Conv2D with height=1
Split protocol: source_group_8_1_1
Source-label overlap between train/val/test: 0
Train clips: 6,996
Validation clips: 866
Test clips: 870
```

Selection rule:

- `Validation-selected test accuracy` is the primary number because the
  checkpoint is selected by validation accuracy, not by test accuracy.
- `Final checkpoint` and `last-2 ensemble` are diagnostic/reference numbers.
  They are useful signals, but they should not replace validation-guided
  model selection unless the final epoch rule is fixed before training.

Compute metric:

- MAC values are Conv/Linear MACs per 4-second clip.
- FLOPs are approximately `2 x MAC` if multiply and add are counted separately.

## Baseline Before This Sweep

Previous best validation-protocol baseline:

```text
Config: configs/kv260_dsafe_ds1d_weakclass_val.json
Params: 82,474
MAC/clip: 61.84M
Best validation accuracy: 70.67%
Validation-selected test accuracy: 76.21%
Final train accuracy: 89.72%
```

The baseline was already overfitting: train accuracy was high, while validation
stayed near 70%. This means simply adding epochs is not the right next move.

## Five Tested Directions

| ID | Config | Main change | Params | MAC/clip | Final train acc | Best val acc | Val-selected test | Final test | Last-2 ensemble |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `configs/kv260_dsafe_ds1d_weakclass_val.json` | avgmax head + weak-class weights | 82,474 | 61.84M | 89.72% | 70.67% | 76.21% | 76.21% | 75.63% |
| 1 | `configs/kv260_ds1d_mixup_ema_val.json` | mixup + EMA + stronger hard-class bias | 82,474 | 61.84M | 60.15% | 67.21% | 76.90% | 78.51% | 78.28% |
| 2 | `configs/kv260_ds1d_focal_mild_val.json` | focal CE + mild class weights + EMA | 82,474 | 61.84M | 86.77% | 71.71% | 73.68% | 75.75% | 76.21% |
| 3 | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` | temporal pyramid avg+max head + mixup + EMA | 101,674 | 61.85M | 64.65% | 72.86% | 79.08% | 79.43% | 79.89% |
| 4 | `configs/kv260_ds1d_multistem_mixup_ema_val.json` | multi-scale waveform stem + mixup + EMA | 82,602 | 63.88M | 62.90% | 70.79% | 72.64% | 73.22% | 73.22% |
| 5 | `configs/kv260_ds1d_combined_priority_val.json` | pyramid + multi-scale stem + focal + mixup + EMA | 101,802 | 63.90M | 62.10% | 69.40% | 75.52% | 75.52% | 74.60% |

## What Was Found

### 1. Mixup + EMA alone improves test signal but underfits validation

The first direction kept the 82K-parameter model and added mixup, EMA, and
stronger hard-class weighting/sampling.

Result:

```text
Best val: 67.21%
Validation-selected test: 76.90%
Final test: 78.51%
Final train: 60.15%
```

Interpretation:

- Mixup regularizes aggressively; the low final train accuracy is expected but
  also indicates underfit.
- The final checkpoint is better than the validation-selected checkpoint on
  test, but validation itself is weaker than baseline. That means this setup is
  not trustworthy as the primary selection strategy.
- It helped `car_horn` and `jackhammer`, but validation did not track that
  improvement well enough.

### 2. Focal loss improves validation but hurts transfer to test

The second direction used mild focal cross entropy and EMA without the weighted
sampler.

Result:

```text
Best val: 71.71%
Validation-selected test: 73.68%
Final test: 75.75%
```

Interpretation:

- Focal loss made validation look better, but test accuracy dropped.
- The likely reason is boundary shift: focal loss emphasizes hard examples, but
  in this source-group split, hard validation examples are not always the same
  failure modes as hard test examples.
- Per-class output confirms the issue: final `car_horn` fell to 60.00% and
  `jackhammer` fell to 59.18%.

### 3. Temporal pyramid pooling is the strongest improvement

The third direction replaced global avg+max with pyramid avg+max pooling over
temporal bins `[1, 2, 4]`.

Result:

```text
Best val: 72.86%
Validation-selected test: 79.08%
Final test: 79.43%
Last-2 ensemble: 79.89%
Params: 101,674
MAC/clip: 61.85M
```

Why this helps:

- A single global average/max vector loses where evidence occurs in the 4-second
  clip.
- Pyramid pooling keeps coarse temporal regions: whole clip, halves, and
  quarters.
- This gives the classifier more information about intermittent events and
  local texture changes without adding convolutional MACs.

The compute effect is favorable:

```text
Baseline params: 82,474 -> 101,674
Baseline MAC: 61.84M -> 61.85M
```

The parameter increase is in the classifier head. Conv/Linear MACs barely
change because the added pooling mostly changes feature aggregation and the
final small linear layer.

### 4. Multi-scale stem alone is not the right lever

The fourth direction replaced the single waveform stem with parallel kernels
`15`, `31`, and `63`.

Result:

```text
Best val: 70.79%
Validation-selected test: 72.64%
Final test: 73.22%
```

Interpretation:

- Early multi-scale waveform filters did not improve generalization.
- The biggest failure was `jackhammer`, which fell to 46.94% in the final
  checkpoint.
- This suggests the bottleneck is not only low-level waveform scale. The model
  needs better temporal aggregation and class separation later in the network.

### 5. Combining all ideas made the model worse

The fifth direction combined pyramid pooling, multi-scale stem, focal loss,
mixup, and EMA.

Result:

```text
Best val: 69.40%
Validation-selected test: 75.52%
Final test: 75.52%
```

Interpretation:

- More techniques together did not help.
- Focal loss and multi-scale stem reduced the benefit of pyramid pooling.
- This is a useful negative result: the next step should not be a large bundle
  of changes. Keep the strongest component, then tune one variable at a time.

## Best Model Per-Class Result

Best current direction:

```text
configs/kv260_ds1d_pyramid_mixup_ema_val.json
```

Final checkpoint per-class test accuracy:

| Class | Support | Correct | Accuracy | Main remaining confusions |
|---|---:|---:|---:|---|
| air_conditioner | 100 | 65 | 65.00% | children_playing 19, engine_idling 6 |
| car_horn | 40 | 30 | 75.00% | dog_bark 3, children_playing 2 |
| children_playing | 100 | 70 | 70.00% | street_music 11, siren 8 |
| dog_bark | 100 | 68 | 68.00% | siren 10, children_playing 7 |
| drilling | 100 | 90 | 90.00% | jackhammer 6 |
| engine_idling | 100 | 85 | 85.00% | children_playing 9 |
| gun_shot | 37 | 37 | 100.00% | none |
| jackhammer | 98 | 89 | 90.82% | drilling 3, air_conditioner 2 |
| siren | 93 | 79 | 84.95% | car_horn 7, children_playing 3 |
| street_music | 102 | 78 | 76.47% | children_playing 15, dog_bark 5 |

Comparison to baseline:

| Class | Baseline final | Pyramid final | Change |
|---|---:|---:|---:|
| air_conditioner | 65.00% | 65.00% | +0.00 |
| car_horn | 75.00% | 75.00% | +0.00 |
| children_playing | 77.00% | 70.00% | -7.00 |
| dog_bark | 66.00% | 68.00% | +2.00 |
| drilling | 88.00% | 90.00% | +2.00 |
| engine_idling | 85.00% | 85.00% | +0.00 |
| gun_shot | 100.00% | 100.00% | +0.00 |
| jackhammer | 65.31% | 90.82% | +25.51 |
| siren | 86.02% | 84.95% | -1.07 |
| street_music | 69.61% | 76.47% | +6.86 |

Key class-level conclusion:

- `jackhammer` is no longer the main blocker in the pyramid model.
- The current blockers are `air_conditioner`, `children_playing`, and
  `dog_bark`.
- `street_music` improved but still confuses with `children_playing`.
- The human/background mixture classes are now more important than the
  mechanical texture classes.

## Implementation Changes In This Sweep

Code changes:

- `src/training/trainer.py`
  - added mixup support inside the training loop;
  - added optional EMA update after optimizer steps.
- `train.py`
  - added `FocalLoss`;
  - added `ModelEMA`;
  - added config fields for `mixup`, `ema`, `pool_bins`, and `stem_type`;
  - saved EMA checkpoints when EMA validation is enabled.
- `src/models/kv260_ds1d.py`
  - added `MultiScaleStem2dH1`;
  - added `pyramid_avgmax` pooling with configurable temporal bins;
  - kept all convolutions as Conv2D-H1, so the model is still logical 1D-CNN.
- `tools/analyze_experiment.py`
  - added support for `pool_bins` and `stem_type`;
  - fixed `--eval_all_cycles` validation preload so validation analysis works
    even when `--eval_train` is not used.

## Why Accuracy Is Still Far From 90%

The current failure is not parameter count alone.

Evidence:

```text
Baseline final train accuracy: 89.72%
Pyramid final train accuracy with mixup: 64.65%
Best validation accuracy so far: 72.86%
Best validation-selected test accuracy so far: 79.08%
```

Interpretation:

- Baseline can fit the training split, but source-independent validation is much
  lower. That points to generalization across source groups.
- Mixup+EMA reduces overfit but can also underfit if the strength is too high.
- Pyramid pooling improves temporal evidence retention, but the network still
  has limited capacity for broad human/background classes.
- `air_conditioner`, `children_playing`, and `dog_bark` require better
  separation of background mixtures, not only more jackhammer weighting.

## Recommended Next Experiments

The next path should keep the pyramid head and change one thing at a time.

1. **Pyramid without aggressive mixup**
   - Keep `pyramid_avgmax`.
   - Try mixup `alpha=0.10`, `prob=0.30-0.50`, or disable mixup.
   - Evaluate raw model and EMA separately, because EMA can lag early in
     training.

2. **Pyramid width search under the KV260 budget**
   - Try `width_mult=1.25` and `1.5`.
   - Target should remain below about `200M MAC/clip`.
   - This can improve `children_playing`, `dog_bark`, and `air_conditioner`
     because those classes need more feature diversity than the 82K model has.

3. **Class-pair targeted augmentation**
   - Focus on pairs:
     - `children_playing` vs `street_music`;
     - `dog_bark` vs `siren/children_playing`;
     - `air_conditioner` vs `engine_idling/children_playing`.
   - Add mild random EQ/band filtering or colored noise before adding more
     class weights.

4. **Loss tuning after pyramid is stable**
   - Do not use focal loss as the default yet.
   - If used, try lower `gamma` such as `0.75-1.0` and compare validation
     transfer to test.

5. **Quantization-aware training after accuracy improves**
   - These five runs do not yet apply true QAT.
   - Current optimization is deployment-aware architecture plus training-time
     regularization.
   - QAT should be added after the float pyramid model is stronger, otherwise
     quantization will preserve a weak model rather than create a 90% model.

## Reproducible Commands

Best current run:

```bash
python train.py --fold 1 --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --exp_name local_priority_pyramid_mixup_ema_50ep --epochs 50 --batch_size 64
python tools/analyze_experiment.py --exp_dir experiments/local_priority_pyramid_mixup_ema_50ep/fold_1 --fold 1 --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --eval_all_cycles --eval_modes
```

Full sweep configs:

```text
configs/kv260_ds1d_mixup_ema_val.json
configs/kv260_ds1d_focal_mild_val.json
configs/kv260_ds1d_pyramid_mixup_ema_val.json
configs/kv260_ds1d_multistem_mixup_ema_val.json
configs/kv260_ds1d_combined_priority_val.json
```
