# Dataset card (paper-oriented)

## Primary: UrbanSound8K

| Item | Value |
|---|---|
| Clips | 8 732 |
| Classes | 10 environmental sound categories |
| Official folds | 10 predefined folders |
| Source key | Freesound `fsID` (many clips share one recording source) |
| Project sample rate | 16 kHz mono |
| Clip length | 4.0 s → **64 000** samples |
| Framing (MAIN models) | **1 frame / clip** (full-clip) |

Official reference: [UrbanSound8K dataset page](https://urbansounddataset.weebly.com/urbansound8k.html).

### Why structure matters

Same `fsID` in train and test allows models to memorize recording fingerprints rather than class acoustics. Headline claims in this repository therefore use **source-safe** or **official-fold** protocols with explicit overlap checks (`source_label_overlap_train_test.count == 0` for source-group).

### Preprocess pipeline

```text
WAV → mono resample 16 kHz → pad/crop 4 s → float tensor [1, 64000]
```

No mel/MFCC is required for MAIN waveform students. Log-mel is used only for secondary log-mel nets and for the AST teacher front-end.

### Seed / split protocols (paper names)

| Paper name | Short | Internal key | Structure | Seed | Trained? |
|---|---|---|---|---:|---|
| **Source-Disjoint Protocol** | **SDP 8-1-1** | `source_group_8_1_1` + `fsid_classid_balanced_v1` | 8/1/1 source buckets | **83** | **Yes** (Tracks 1–3) |
| **Official-Fold Protocol** | **OFP 8-1-1** | `clean_8_1_1` | test fold1, val fold2, train 3–10 | 83 | **Yes** (OFP Baseline + MC-ISR) |
| **Literature 9+1 Protocol** | **L91** | `paper_9_1` | 9 train folds / no val / 1 test | — | Optional only |

Figures: `fig01_pipeline_framing_fullclip.png`, `fig01b_protocol_seed_status.png`. Full map: [NAMING.md](NAMING.md).

### Class names (label order)

0 air_conditioner · 1 car_horn · 2 children_playing · 3 dog_bark · 4 drilling ·  
5 engine_idling · 6 gun_shot · 7 jackhammer · 8 siren · 9 street_music

---

## Secondary datasets (paper scope — **not yet seeded**)

| Dataset | Role | Loader | Seeded | Trained |
|---|---|---|---|---|
| **ESC-50** | environmental multi-class comparison (50 classes, 5-fold) | **not implemented** | **No** | **No** |
| **Speech Commands** (subset) | short-clip / edge / KV260 narrative | **not implemented** | **No** | **No** |

These appear in the paper plan as follow-on experiments. Do not report accuracy for them until loaders, splits, and metrics are committed.

---

## Analysis standard

After every trained fold, require:

1. Protocol + seed + config path recorded  
2. Primary metric filled (`test_acc_best_val_model` when val exists)  
3. Source overlap summary for US8K source-group  
4. Confusion / worst class noted  
5. Comparison to the correct baseline for that protocol  

Canonical checklists: `docs/data/ANALYSIS_STANDARD.md`, `docs/data/SPLIT_PROTOCOL.md`.
