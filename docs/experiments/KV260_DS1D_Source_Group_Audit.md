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

