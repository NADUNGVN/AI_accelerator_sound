# KV260 DS1D Layer Depth Check

Date: 2026-07-19

## Purpose

Check whether adding layers to the current best KV260 1D-CNN improves
source-safe UrbanSound8K accuracy while staying inside the board-oriented
budget.

Baseline reference:

```text
config: configs/kv260_ds1d_pyramid_mixup_ema_val.json
model: kv260_audio_net_ds1d
params: 101,674
MAC/clip: 61,854,400
FLOPs convention: 2 * MAC = 123,708,800 FLOPs/clip
fold-1 final: 79.43%
fold-1 ensemble: 79.89%
folds 1-3 mean final: 71.88%
folds 1-3 mean ensemble: 72.30%
```

## Implemented Variants

### LateRes2

Config:

```text
configs/kv260_ds1d_pyramid_lateres2_mixup_ema_val.json
```

Architecture change:

```text
Keep the baseline channel plan and all original DS blocks.
Add 2 residual depthwise-separable Conv2D-H1 blocks at the final 160-channel
late stage after temporal downsampling.
```

Budget:

```text
params: 157,994
MAC/clip: 75,614,400
FLOPs convention: 2 * MAC = 151,228,800 FLOPs/clip
```

### Deep

Config:

```text
configs/kv260_ds1d_deep_pyramid_mixup_ema_val.json
```

Architecture change:

```text
Use a deeper residual DS1D model with late residual blocks and final channels
increased from 160 to 192.
```

Budget:

```text
params: 196,074
MAC/clip: 104,874,880
FLOPs convention: 2 * MAC = 209,749,760 FLOPs/clip
```

Both variants remain below the configured limits:

```text
max_params: 300,000
max_macs_per_clip: 300,000,000
```

## Commands

Smoke tests:

```bash
python train.py --fold 1 --config configs/kv260_ds1d_pyramid_lateres2_mixup_ema_val.json --exp_name smoke_lateres2_local --epochs 1 --max_train_clips 80 --max_val_clips 20 --max_test_clips 20
python train.py --fold 1 --config configs/kv260_ds1d_deep_pyramid_mixup_ema_val.json --exp_name smoke_deep_layers_local --epochs 1 --max_train_clips 80 --max_val_clips 20 --max_test_clips 20
```

Fold-1 checks:

```bash
python tools/run_multifold.py --config configs/kv260_ds1d_pyramid_lateres2_mixup_ema_val.json --exp_name local_lateres2_f1_50ep --folds 1 --epochs 50 --analyze --eval_modes
python tools/run_multifold.py --config configs/kv260_ds1d_deep_pyramid_mixup_ema_val.json --exp_name local_deep_layers_f1_50ep --folds 1 --epochs 50 --analyze --eval_modes
```

Source audits:

```bash
python tools/audit_source_groups.py --exp_name local_lateres2_f1_50ep --folds 1 --classes air_conditioner,engine_idling,jackhammer,drilling,children_playing,street_music,siren --min_support 5 --max_groups 15 --max_spectrogram_clips 6
python tools/audit_source_groups.py --exp_name local_deep_layers_f1_50ep --folds 1 --classes air_conditioner,engine_idling,jackhammer,drilling,children_playing,street_music,siren --min_support 5 --max_groups 15 --max_spectrogram_clips 6
```

## Fold-1 Results

| Model | Params | MAC/clip | Final test | Last-2 ensemble | Worst class |
|---|---:|---:|---:|---:|---|
| Baseline | 101,674 | 61,854,400 | 79.43% | 79.89% | air_conditioner 65.00% |
| LateRes2 | 157,994 | 75,614,400 | 75.52% | 75.86% | jackhammer 46.94% |
| Deep | 196,074 | 104,874,880 | 71.72% | 72.41% | air_conditioner 36.00% |

Per-class fold-1 comparison:

| Class | Baseline | LateRes2 | LateRes2 delta | Deep | Deep delta |
|---|---:|---:|---:|---:|---:|
| air_conditioner | 65.00% | 66.00% | +1.00 | 36.00% | -29.00 |
| car_horn | 75.00% | 70.00% | -5.00 | 72.50% | -2.50 |
| children_playing | 70.00% | 77.00% | +7.00 | 76.00% | +6.00 |
| dog_bark | 68.00% | 68.00% | +0.00 | 77.00% | +9.00 |
| drilling | 90.00% | 92.00% | +2.00 | 81.00% | -9.00 |
| engine_idling | 85.00% | 87.00% | +2.00 | 85.00% | +0.00 |
| gun_shot | 100.00% | 100.00% | +0.00 | 100.00% | +0.00 |
| jackhammer | 90.82% | 46.94% | -43.88 | 45.92% | -44.90 |
| siren | 84.95% | 82.80% | -2.15 | 90.32% | +5.38 |
| street_music | 76.47% | 77.45% | +0.98 | 72.55% | -3.92 |

## Source Audit Finding

Both added-layer variants heavily damage the same fold-1 jackhammer source:

| Model | Class | fsID | Accuracy | Main confusion |
|---|---|---:|---:|---|
| LateRes2 | jackhammer | 177537 | 4/45 | air_conditioner 36, drilling 5 |
| Deep | jackhammer | 177537 | 1/45 | air_conditioner 39, engine_idling 5 |

LateRes2 still fails the known weak air_conditioner sources:

```text
air_conditioner fsID 146690: 0/25, predicted children_playing 25
air_conditioner fsID 80589: 0/5, predicted car_horn 5
```

Deep makes the air_conditioner problem worse:

```text
air_conditioner final accuracy: 36.00%
air_conditioner fsID 146690: 0/25, predicted children_playing 23, engine_idling 2
air_conditioner fsID 146845: 2/32, mostly predicted engine_idling
```

## Decision

Do not promote either added-layer model to folds 2-3 or 10-fold testing.

Adding layers is feasible for KV260 from a budget perspective, but these two
variants reduce source-safe accuracy. The failure mode is not parameter
shortage. The added layers shift the decision boundary toward stationary
texture cues and severely damage jackhammer source generalization.

Current best model remains:

```text
configs/kv260_ds1d_pyramid_mixup_ema_val.json
params: 101,674
MAC/clip: 61,854,400
folds 1-3 mean final: 71.88%
folds 1-3 mean ensemble: 72.30%
```

## Next Direction

Depth alone is not the right next step. The next candidate should improve
representation without changing the late decision boundary as aggressively.
Prefer a controlled dual-feature or auxiliary-target experiment over simply
adding more convolutional depth.

