# KV260 DS1D Source Group Audit

Date: 2026-07-19

## Scope

This audit checks why the current source-safe 1D-CNN baseline is stuck around
72% multi-fold accuracy even though fold 1 can reach about 79%.

Baseline audited:

```text
experiment: local_multifold_pyramid_base_f1_f3_50ep
config: configs/kv260_ds1d_pyramid_mixup_ema_val.json
model: kv260_audio_net_ds1d
params: 101,674
MAC/clip: 61,854,400
protocol: source_group_8_1_1
folds: 1-3
prediction set: last_snapshot_predictions
```

The detailed local report is generated under:

```text
experiments/local_multifold_pyramid_base_f1_f3_50ep/source_audit/source_group_audit.md
```

`experiments/` is ignored by git, so regenerate it with:

```bash
python tools/audit_source_groups.py --exp_name local_multifold_pyramid_base_f1_f3_50ep --folds 1-3 --classes air_conditioner,engine_idling,jackhammer,street_music,drilling,children_playing,siren --min_support 5 --max_groups 20 --max_spectrogram_clips 6
```

## Result Summary

Mean fold accuracy for this baseline:

| Metric | Accuracy |
|---|---:|
| Mean final test, folds 1-3 | 71.88% |
| Mean last-2 ensemble, folds 1-3 | 72.30% |
| Fold 1 final / ensemble | 79.43% / 79.89% |
| Fold 2 final / ensemble | 67.32% / 68.82% |
| Fold 3 final / ensemble | 68.89% / 68.20% |

Per-class accuracy across folds 1-3:

| Class | Total | Errors | Accuracy |
|---|---:|---:|---:|
| air_conditioner | 300 | 174 | 42.00% |
| car_horn | 118 | 16 | 86.44% |
| children_playing | 300 | 89 | 70.33% |
| dog_bark | 300 | 64 | 78.67% |
| drilling | 300 | 61 | 79.67% |
| engine_idling | 300 | 120 | 60.00% |
| gun_shot | 112 | 0 | 100.00% |
| jackhammer | 298 | 65 | 78.19% |
| siren | 277 | 47 | 83.03% |
| street_music | 302 | 97 | 67.88% |

## Error Concentration

The main failure is not random clip noise. Errors are concentrated in a small
number of held-out source groups.

| Class | Errors | Top-5 source errors | Top-10 source errors | Source groups with errors |
|---|---:|---:|---:|---:|
| air_conditioner | 174 | 117 (67.2%) | 162 (93.1%) | 15 |
| engine_idling | 120 | 87 (72.5%) | 107 (89.2%) | 18 |
| jackhammer | 65 | 64 (98.5%) | 65 (100.0%) | 6 |
| drilling | 61 | 47 (77.0%) | 57 (93.4%) | 14 |
| siren | 47 | 32 (68.1%) | 44 (93.6%) | 13 |
| children_playing | 89 | 38 (42.7%) | 62 (69.7%) | 26 |
| street_music | 97 | 31 (32.0%) | 56 (57.7%) | 31 |

Worst audited groups:

| Fold | Class | fsID | Accuracy | Main confusions | Audio means |
|---:|---|---:|---:|---|---|
| 2 | engine_idling | 94632 | 0/31 | street_music 12, air_conditioner 7, drilling 7, jackhammer 5 | rms -27.4 dB, centroid 1432 Hz |
| 3 | air_conditioner | 74677 | 0/31 | street_music 21, children_playing 7, jackhammer 3 | rms -26.7 dB, centroid 1563 Hz |
| 1 | air_conditioner | 146690 | 0/25 | children_playing 19, engine_idling 6 | rms -16.6 dB, centroid 1204 Hz |
| 3 | air_conditioner | 74507 | 0/25 | jackhammer 25 | rms -23.1 dB, centroid 1557 Hz |
| 2 | air_conditioner | 162103 | 0/18 | drilling 18 | rms -25.7 dB, centroid 1717 Hz |
| 2 | drilling | 180937 | 0/16 | jackhammer 15, air_conditioner 1 | rms -17.3 dB, centroid 2660 Hz |
| 3 | drilling | 77751 | 0/15 | jackhammer 15 | rms -23.7 dB, centroid 1038 Hz |
| 3 | air_conditioner | 202516 | 0/12 | engine_idling 12 | rms -26.8 dB, centroid 919 Hz |
| 3 | engine_idling | 177592 | 0/11 | siren 6, children_playing 2, street_music 2, air_conditioner 1 | rms -23.5 dB, centroid 1353 Hz |
| 3 | jackhammer | 132016 | 0/9 | children_playing 5, air_conditioner 4 | rms -28.0 dB, centroid 1262 Hz |

## Findings

1. The 72% ceiling is primarily a source-domain generalization problem.
   Source-safe splitting prevents leakage, so the test set contains `fsID`
   groups whose sound texture is not covered well by the training groups.

2. `air_conditioner` is not one homogeneous class. Different held-out sources
   collapse into different labels: `74507 -> jackhammer`, `74677 -> street_music`
   or `children_playing`, `162103 -> drilling`, and `202516 -> engine_idling`.
   A single global class weight cannot solve these opposite confusions.

3. The drilling/jackhammer boundary is a separate failure mode. Multiple
   drilling sources are predicted as jackhammer, while one jackhammer source
   is predicted as children_playing or air_conditioner. This explains why broad
   SupCon improved one air_conditioner source but damaged jackhammer.

4. Some failed groups have very low RMS or strong stationary tonal bands.
   This indicates that amplitude, background dominance, and source texture are
   part of the problem. More epochs alone will keep fitting the seen sources
   and will not reliably recover unseen source groups.

5. The 200-epoch checks confirm this diagnosis. Fold 1 at 200 epochs did not
   beat the 50-epoch baseline, and the paper-LR 200-epoch run was worse.

## Decision

Do not spend more time on plain longer training, broad all-class SupCon, or
width-only scaling until the source-domain failure is addressed.

The next experiment should optimize during training for source robustness:

```text
Keep model under 300k params.
Keep MAC/clip in the KV260-safe budget already established.
Use source-aware sampling and targeted confusion-pair regularization.
Reject a candidate if it improves one weak class while collapsing jackhammer.
```

## Next Experiment Direction

Recommended next run:

```text
source-aware hard-group training
```

Implementation idea:

1. Build a source-balanced sampler so mini-batches do not overfit dominant
   source textures.
2. Add targeted augmentation for the audited weak boundaries:
   `air_conditioner <-> engine_idling`, `air_conditioner <-> jackhammer`,
   `air_conditioner <-> drilling`, `drilling <-> jackhammer`,
   and `children_playing/street_music/siren`.
3. Keep the current `kv260_audio_net_ds1d` family first. Do not change the
   model architecture until a training-side robustness check shows positive
   movement.
4. Run folds 1-3 for 50 epochs, then only promote to 10 folds if:
   mean final accuracy improves over 71.88%, ensemble improves over 72.30%,
   `air_conditioner` improves without `jackhammer` falling below baseline.

## Local Source-Aware Training Check

Date: 2026-07-19

Machine:

```text
Windows local machine
Python: C:\Users\Dawin\AppData\Local\Programs\Python\Python311\python.exe
Device: cuda
GPU-compatible batch tested: 64 full-clip waveforms
```

Code change:

```text
train.py now supports a top-level source_aware_batch_sampler for CE training.
The sampler no longer requires supervised_contrastive.enabled=true.
```

### Smoke Test

Command:

```bash
python train.py --fold 1 --config configs/kv260_ds1d_pyramid_sourcehard_ce_val.json --exp_name smoke_sourcehard_local --epochs 1 --max_train_clips 80 --max_val_clips 20 --max_test_clips 20
```

Result:

```text
Device: cuda
Source-label overlap train/test: 0
Source-label overlap train/val: 0
Source-label overlap val/test: 0
Params: 101,674
MAC/clip: 61,854,400
Status: DataLoader, source sampler, loss, hard-negative margin, EMA, and eval all run successfully.
```

### Source-Hard CE

Config:

```text
configs/kv260_ds1d_pyramid_sourcehard_ce_val.json
```

Purpose:

```text
Use source-aware batches, class multipliers for weak classes, and targeted
hard-negative margin on audited confusion pairs.
```

Command:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_sourcehard_ce_val.json --exp_name local_sourcehard_ce_f1_50ep --folds 1 --epochs 50 --analyze --eval_modes
```

Fold-1 result:

| Run | Final test | Last-2 ensemble | Worst class |
|---|---:|---:|---|
| Baseline | 79.43% | 79.89% | air_conditioner 65.00% |
| Source-hard CE | 76.21% | 77.01% | jackhammer 60.20% |

Fold-1 per-class delta against baseline:

| Class | Baseline | Source-hard | Delta |
|---|---:|---:|---:|
| air_conditioner | 65.00% | 69.00% | +4.00 |
| car_horn | 75.00% | 77.50% | +2.50 |
| children_playing | 70.00% | 65.00% | -5.00 |
| dog_bark | 68.00% | 75.00% | +7.00 |
| drilling | 90.00% | 81.00% | -9.00 |
| engine_idling | 85.00% | 86.00% | +1.00 |
| gun_shot | 100.00% | 100.00% | +0.00 |
| jackhammer | 90.82% | 60.20% | -30.61 |
| siren | 84.95% | 87.10% | +2.15 |
| street_music | 76.47% | 77.45% | +0.98 |

Decision:

```text
Reject source-hard CE in this form. It confirms that targeted pressure can
recover air_conditioner slightly, but the hard-negative pair/multiplier setup
damages jackhammer too much. Do not run folds 2-3 or 10-fold promotion for this
config.
```

### Source-Balanced CE

Config:

```text
configs/kv260_ds1d_pyramid_sourcebalance_ce_val.json
```

Purpose:

```text
Isolate the source-aware sampler by removing hard-negative margin and removing
strong weak-class sampler multipliers.
```

Commands:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_sourcebalance_ce_val.json --exp_name local_sourcebalance_ce_f1_50ep --folds 1 --epochs 50 --analyze --eval_modes
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_sourcebalance_ce_val.json --exp_name local_sourcebalance_ce_f1_50ep --folds 1-3 --epochs 50 --analyze --eval_modes --skip_existing
python tools/audit_source_groups.py --exp_name local_sourcebalance_ce_f1_50ep --folds 1-3 --classes air_conditioner,engine_idling,jackhammer,drilling,children_playing,street_music,siren --min_support 5 --max_groups 20 --max_spectrogram_clips 6
```

Folds 1-3 result:

| Run | Mean final | Mean ensemble | Worst-class mean |
|---|---:|---:|---:|
| Baseline | 71.88% | 72.30% | 42.00% |
| Source-balanced CE | 70.30% | 69.46% | 42.33% |

Per-fold result:

| Fold | Baseline final | Source-balanced final | Delta |
|---:|---:|---:|---:|
| 1 | 79.43% | 79.31% | -0.11 |
| 2 | 67.32% | 64.90% | -2.42 |
| 3 | 68.89% | 66.70% | -2.19 |

Per-class mean delta:

| Class | Baseline | Source-balanced | Delta |
|---|---:|---:|---:|
| air_conditioner | 42.00% | 43.33% | +1.33 |
| car_horn | 86.54% | 91.60% | +5.06 |
| children_playing | 70.33% | 66.33% | -4.00 |
| dog_bark | 78.67% | 77.33% | -1.33 |
| drilling | 79.67% | 74.00% | -5.67 |
| engine_idling | 60.00% | 59.67% | -0.33 |
| gun_shot | 100.00% | 100.00% | +0.00 |
| jackhammer | 78.22% | 73.52% | -4.69 |
| siren | 83.03% | 85.93% | +2.91 |
| street_music | 67.77% | 64.12% | -3.65 |

Source audit after source-balanced CE still shows the same held-out source
collapses:

| Fold | Class | fsID | Accuracy | Main confusion |
|---:|---|---:|---:|---|
| 2 | engine_idling | 94632 | 0/31 | jackhammer, air_conditioner, street_music, drilling |
| 3 | air_conditioner | 74677 | 0/31 | street_music 31 |
| 1 | air_conditioner | 146690 | 0/25 | engine_idling 25 |
| 3 | air_conditioner | 74507 | 0/25 | jackhammer 25 |
| 2 | air_conditioner | 162103 | 0/18 | drilling 18 |

Decision:

```text
Reject source-balanced CE as an improvement candidate. It is safe enough to keep
as an ablation tool, but it does not solve source-domain generalization and does
not beat the baseline on folds 1-3.
```

Current conclusion:

```text
The failure is deeper than batch composition. The model still lacks features
that separate unseen source textures inside air_conditioner, engine_idling,
drilling, jackhammer, children_playing, siren, and street_music. The next useful
direction should change the representation or training target more directly,
while protecting jackhammer and drilling with rejection criteria.
```
