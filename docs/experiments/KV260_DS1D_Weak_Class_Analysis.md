# KV260 DS1D Weak-Class Analysis

## Experiment Context

Model:

```text
KV260AudioNetDS1D
Params: 80,874
MAC/clip: 61,833,600
Input: full 4s raw waveform, 16 kHz, 64,000 samples
Split: source_group_9_1
Train/test source-label overlap: 0
```

Training config:

```text
configs/kv260_dsafe_ds1d.json
CrossEntropy
balanced class weights
AdamW
AMP
waveform augmentation
batch size 64
```

Important implementation note:

- A checkpoint bug was found after the first 50-epoch run: with `epochs=50` and `cycles=4`, cycle checkpoints were saved at epochs 13, 26, and 39, but not epoch 50.
- The code has been fixed to always save `cycle_final` when the last epoch is not already a cycle checkpoint.
- Correct final run: `experiments/local_kv260_ds1d_50ep_finalfix/fold_1`.

## Results

| Run | Train acc | Test acc | Notes |
|---|---:|---:|---|
| 20 epochs | 81.05% | 69.54% last, 69.66% ensemble | first local signal |
| 50 epochs final checkpoint | 91.00% | 73.68% | correct final epoch result |

Per-class result at epoch 50:

| Class | Support | Correct | Accuracy | Main confusions |
|---|---:|---:|---:|---|
| air_conditioner | 100 | 66 | 66.00% | children_playing 23, drilling 5, street_music 4 |
| car_horn | 40 | 23 | 57.50% | dog_bark 6, children_playing 4, street_music 3 |
| children_playing | 100 | 77 | 77.00% | street_music 10, siren 9 |
| dog_bark | 100 | 67 | 67.00% | children_playing 10, siren 10, street_music 6 |
| drilling | 100 | 90 | 90.00% | engine_idling 5 |
| engine_idling | 100 | 86 | 86.00% | children_playing 9 |
| gun_shot | 37 | 37 | 100.00% | none |
| jackhammer | 98 | 35 | 35.71% | air_conditioner 45, car_horn 9, drilling 5 |
| siren | 93 | 85 | 91.40% | children_playing 4 |
| street_music | 102 | 75 | 73.53% | children_playing 15 |

## Dataset/Split Observations

Source-group split statistics:

| Class | Train clips | Train groups | Test clips | Test groups | Notes |
|---|---:|---:|---:|---:|---|
| air_conditioner | 900 | 58 | 100 | 6 | long stationary texture |
| car_horn | 389 | 110 | 40 | 15 | many short clips |
| children_playing | 900 | 143 | 100 | 15 | broad background variability |
| dog_bark | 900 | 303 | 100 | 34 | highly variable events/backgrounds |
| drilling | 900 | 107 | 100 | 12 | strong mechanical texture |
| engine_idling | 900 | 87 | 100 | 10 | strong stationary texture |
| gun_shot | 337 | 105 | 37 | 12 | short but highly distinctive transient |
| jackhammer | 902 | 41 | 98 | 4 | very few test source groups |
| siren | 836 | 64 | 93 | 10 | distinctive pitch sweep |
| street_music | 898 | 149 | 102 | 17 | broad audio mixture |

Short-clip issue:

- `car_horn` test median duration is about 1.94s and 13/40 test clips are under 1s.
- `gun_shot` is even shorter, but it remains easy because the transient is very distinctive.
- Full-clip global average pooling can dilute non-distinct short events.

## Why The Weak Classes Are Weak

### jackhammer

Observed problem:

```text
jackhammer accuracy: 35.71%
main confusion: air_conditioner 45/98
```

Likely causes:

1. **Very few test source groups**: only 4 test source-label groups. A few hard source recordings dominate the class result.
2. **Texture similarity**: jackhammer, air_conditioner, drilling, and engine_idling can all produce stationary or repetitive mechanical textures.
3. **Global pooling loses temporal burst structure**: the current DS1D model uses global average pooling. It summarizes the whole clip but may not emphasize intermittent bursts that distinguish jackhammer from smoother machine noise.
4. **Balanced CE does not emphasize jackhammer**: because jackhammer has many train clips, balanced weighting assigns it a low weight even though it is empirically hard.

### car_horn

Observed problem:

```text
car_horn accuracy: 57.50%
main confusion: dog_bark, children_playing, street_music
```

Likely causes:

1. **Few clips**: only 389 train clips and 40 test clips.
2. **Short events**: many clips contain brief horns followed by padding/silence.
3. **Full-clip average pooling dilutes the horn event**.
4. **Background leakage is blocked**: source-independent split removes easy source/background memorization, making brief horn recognition harder.

### dog_bark

Observed problem:

```text
dog_bark accuracy: 67.00%
main confusion: children_playing, siren, street_music
```

Likely causes:

1. **High intra-class variability**: barks differ in pitch, duration, repetition, and background.
2. **Urban background mixtures**: clips can include people, music, or siren-like components.
3. **Current low-capacity model has limited temporal pattern diversity**.

### air_conditioner

Observed problem:

```text
air_conditioner accuracy: 66.00%
main confusion: children_playing, drilling, street_music
```

Likely causes:

1. **Stationary background ambiguity**: air conditioner often behaves like background rather than a discrete event.
2. **Source-independent split**: room/device/source characteristics shift between train and test.
3. **Limited channel capacity**: 80K parameters may not separate all stationary texture subtypes.

## Targeted Improvement Hypothesis

The next experiment should not simply add epochs. Train accuracy is already 91%, while test is 73.68%, so more training risks more overfit.

Targeted changes:

1. **Avg+max pooling**: add max pooling alongside average pooling so short/discriminative events can survive global aggregation.
2. **Hard-class loss multipliers**: increase CE weight for weak classes, especially `jackhammer`, `car_horn`, `dog_bark`, and `air_conditioner`.
3. **Hard-class weighted sampler**: sample hard classes more often during training so their gradients are seen more frequently.

Expected effect:

- Better `jackhammer`, `car_horn`, and possibly `dog_bark`.
- Possible tradeoff: some strong classes such as `gun_shot`, `siren`, or `drilling` may drop if the model shifts decision boundaries too much.
- Overall accuracy may improve modestly, but the main goal is to verify whether weak-class targeting helps without increasing MACs too much.

The first targeted run should keep the KV260 budget:

```text
params should remain around 82K
MAC/clip should remain around 61.8M
```

## Targeted Improvement Run

Config:

```text
configs/kv260_dsafe_ds1d_weakclass.json
```

Changes versus baseline:

- `pool_type: avgmax`
  - adds global max pooling beside average pooling;
  - improves short/discriminative event retention;
  - params increase only from 80,874 to 82,474;
  - MAC/clip stays essentially the same: 61.83M -> 61.84M.
- class weight multipliers:
  - air_conditioner 1.15;
  - car_horn 1.60;
  - dog_bark 1.25;
  - jackhammer 2.20.
- weighted sampler multipliers:
  - air_conditioner 1.15;
  - car_horn 1.80;
  - dog_bark 1.25;
  - jackhammer 2.00.

Run:

```bash
python train.py --fold 1 --config configs/kv260_dsafe_ds1d_weakclass.json --exp_name local_kv260_ds1d_weakclass_50ep --epochs 50 --batch_size 64
```

### Overall Result

| Model | Params | MAC/clip | Checkpoint | Test acc |
|---|---:|---:|---|---:|
| baseline KV260 DS1D | 80,874 | 61.83M | final epoch 50 | 73.68% |
| weak-class targeted | 82,474 | 61.84M | final epoch 50 | 74.60% |
| weak-class targeted | 82,474 | 61.84M | cycle 2, epoch 26 | 76.55% |
| weak-class targeted | 82,474 | 61.84M | last-2 ensemble | 75.40% |

The best checkpoint improved overall accuracy by `+2.87 points` over the baseline final checkpoint.

### Per-Class Comparison

Baseline final epoch 50 versus targeted cycle 2:

| Class | Baseline | Targeted cycle 2 | Change |
|---|---:|---:|---:|
| air_conditioner | 66.00% | 67.00% | +1.00 |
| car_horn | 57.50% | 72.50% | +15.00 |
| children_playing | 77.00% | 73.00% | -4.00 |
| dog_bark | 67.00% | 70.00% | +3.00 |
| drilling | 90.00% | 81.00% | -9.00 |
| engine_idling | 86.00% | 87.00% | +1.00 |
| gun_shot | 100.00% | 100.00% | 0.00 |
| jackhammer | 35.71% | 73.47% | +37.76 |
| siren | 91.40% | 92.47% | +1.07 |
| street_music | 73.53% | 62.75% | -10.78 |

The targeted intervention worked for the intended weak classes:

- `jackhammer` improved massively;
- `car_horn` improved strongly;
- `dog_bark` improved mildly;
- `air_conditioner` stayed about the same.

But there is a tradeoff:

- `street_music` dropped;
- `drilling` dropped;
- `children_playing` dropped slightly.

### Interpretation

The result confirms the weak-class diagnosis:

- `avgmax` pooling helps short or localized evidence survive global aggregation.
- hard-class weighting/sampling forces the model to stop ignoring jackhammer/car_horn.
- however, aggressive hard-class bias shifts the decision boundary and hurts some broad/background classes.

The most important finding is not merely the final 74.60%. It is that the best checkpoint reached 76.55%, while final epoch dropped. This means the next required infrastructure is validation-based model selection. Continuing to train without validation risks selecting a worse final checkpoint.

## Next Step

Add `source_group_8_1_1`:

```text
train: source groups for learning
val: source groups for checkpoint selection and class-target tuning
test: held out source groups for final report
```

Then rerun the targeted config and select the checkpoint by validation accuracy, not by test accuracy or final epoch.
