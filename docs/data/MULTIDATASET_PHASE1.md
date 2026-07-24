# Phase 1 multi-dataset expansion

**Status:** ESC-50 and Speech Commands server results are now committed. ESC-10 is added as a verification subset loaded from the ESC-50 metadata.

**Branch rule:** use git `main` only.

**Default server:** `CPU-FPGA-GPU` RTX 3090 only; see [`../main/SERVER_POLICY.md`](../main/SERVER_POLICY.md).

## Research basis

| Dataset | Relevant source facts | Project implication |
|---|---|---|
| UrbanSound8K | Official 10 folds; the dataset page warns that random reshuffling can put related samples in train and test and inflate scores. | Keep SDP/OFP results separate; random clip split is a leakage diagnostic only. |
| ESC-10 | 400 clips, 10 classes, 5 s clips; distributed as the `esc10` subset flag inside ESC-50 metadata. | Use the same ESC-50 archive; report as a separate dataset/protocol from ESC-50. |
| ESC-50 | 2,000 clips, 50 classes, 5 s clips, 40 examples/class; official 5 folds keep fragments from the same Freesound source file in one fold. | Use official folds; do not random split. Add validation without breaking fold identity. |
| Speech Commands | Keyword spotting dataset with 10 target words plus unknown/silence handling; TFDS exposes train/validation/test and 12 labels. | Use the official fixed split; do not create ad hoc speaker/file splits for headline numbers. |

Sources checked on 2026-07-23:

- UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html
- ESC-50: https://github.com/karolpiczak/esc-50
- Speech Commands paper: https://arxiv.org/abs/1804.03209
- Speech Commands TFDS: https://www.tensorflow.org/datasets/catalog/speech_commands

## Dataset contracts

| Dataset | Internal protocol | Input contract | Classes | Split contract | Primary metric |
|---|---|---|---:|---|---|
| UrbanSound8K | `source_group_8_1_1` / `clean_8_1_1` / optional `paper_9_1` | 16 kHz mono, 4.0 s, 64,000 samples | 10 | Existing SDP/OFP contracts in [`SPLIT_PROTOCOL.md`](SPLIT_PROTOCOL.md) | `test_acc_best_val_model` when val exists |
| ESC-10 | `esc10_3_1_1_foldk_valnext_v1` | 16 kHz mono, 5.0 s, 80,000 samples | 10 | Filter `meta/esc50.csv` where `esc10` is true; for each test fold `k`, use next fold as validation and remaining 3 folds as train. | Mean/std of validation-selected test accuracy |
| ESC-10 literature side table | `esc10_official_4_1_cv` | same as above | 10 | Train 4 folds, test 1 fold, no validation. Keep separate from the deploy-style val protocol. | Mean/std test accuracy under 5-fold CV |
| ESC-50 | `esc50_3_1_1_foldk_valnext_v1` | 16 kHz mono, 5.0 s, 80,000 samples | 50 | For each test fold `k`, use next fold as validation and remaining 3 folds as train; repeat all 5 test folds. | Mean/std of validation-selected test accuracy |
| ESC-50 literature side table | `esc50_official_4_1_cv` | same as above | 50 | Train 4 folds, test 1 fold, no validation. Keep separate from the deploy-style val protocol. | Mean/std test accuracy under 5-fold CV |
| Speech Commands subset | `speech_commands_v2_official12` | 16 kHz mono, 1.0 s, 16,000 samples | 12 | Use official train/validation/test split. Preserve unknown and silence handling. | Validation-selected test accuracy; macro-F1 as secondary |

Do not compare `esc10_3_1_1_foldk_valnext_v1` or `esc50_3_1_1_foldk_valnext_v1` directly to papers that report official 4-train/1-test CV; the training data and checkpoint-selection rule differ.

Speech Commands parser currently targets the raw v0.02 folder layout with `validation_list.txt`, `testing_list.txt`, word subfolders, and `_background_noise_`. Class order follows TFDS: `down, go, left, no, off, on, right, stop, up, yes, _silence_, _unknown_`. Background-noise files are segmented into 1 s silence windows with 0.5 s hop, matching the TFDS builder behavior.

## Committed Phase 1 Results

| Dataset | Experiment | Protocol | Result status |
|---|---|---|---|
| Speech Commands | `speech_commands_phase1_official12_dsconv2dh1_30ep` | `speech_commands_v2_official12` | Best-val test `90.93%`; final test `91.07%`; ensemble `90.81%`. |
| ESC-50 | `esc50_phase1_dsconv2dh1_5fold_120ep` | `esc50_3_1_1_foldk_valnext_v1` | Validation-selected test mean `43.90% +/- 1.85%`; final mean `43.20% +/- 2.56%`; ensemble mean `43.60% +/- 2.60%`. |
| ESC-10 | pending | `esc10_3_1_1_foldk_valnext_v1` | Added as verification dataset; run full 5 folds before reporting. |

## Phase 1 gates

Before any full training run on a new dataset:

1. Dataset availability check: root path, metadata file, sample count, class count.
2. Loader smoke: one batch train/val/test, tensor shape, dtype, label range.
3. Split fingerprint: fold/split IDs, counts per class, and no accidental overlap in the dataset's source key where available.
4. Config smoke: 1 epoch or tiny-clips smoke with deterministic experiment name.
5. Full first split: one canonical fold/split on `CPU-FPGA-GPU`.
6. Analysis: metrics, confusion, worst classes, and comparison only to the matching dataset/protocol baseline.

Only after all gates pass should Phase 1 expand to all ESC-50 folds or the full Speech Commands official split.

## Approach order

| Step | Purpose | Notes |
|---|---|---|
| P1-S0 loader/registry | Make datasets first-class without changing model claims. | Add dataset-specific metadata and split builders before training. |
| P1-S1 deployable waveform student | Reuse DS-Conv2D-H1 Pyramid with dataset-specific `num_classes` and `clip_seconds`. | This keeps the KV260 story coherent. |
| P1-S2 teacher ceiling | Optional log-mel/AST or other pretrained teacher after S1 is stable. | Teacher is research-only unless distilled into the student. |
| P1-S3 deploy analysis | Params/MACs/latency per dataset head. | Class count changes classifier params, so report complexity per dataset. |

## Server command skeleton

ESC-50 smoke:

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
hostname   # CPU-FPGA-GPU
conda activate sound_env
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

git fetch origin
git checkout main
git pull origin main

python tools/run_multifold.py \
  --config configs/main/student_ds_conv2d_h1_pyramid_esc50_phase1.json \
  --exp_name esc50_phase1_fold1_smoke_1ep \
  --folds 1 \
  --epochs 1 \
  --max_train_clips 64 \
  --max_val_clips 32 \
  --max_test_clips 32 \
  --analyze \
  --eval_modes
```

ESC-10 smoke:

```bash
python tools/run_multifold.py \
  --config configs/main/student_ds_conv2d_h1_pyramid_esc10_phase1.json \
  --exp_name esc10_phase1_fold1_smoke_1ep \
  --folds 1 \
  --epochs 1 \
  --max_train_clips 64 \
  --max_val_clips 32 \
  --max_test_clips 32 \
  --analyze \
  --eval_modes
```

Speech Commands smoke:

```bash
python tools/run_multifold.py \
  --config configs/main/student_ds_conv2d_h1_pyramid_speech_commands_phase1.json \
  --exp_name speech_commands_phase1_official12_smoke_1ep \
  --folds 1 \
  --epochs 1 \
  --max_train_clips 128 \
  --max_val_clips 64 \
  --max_test_clips 64 \
  --analyze \
  --eval_modes
```

Every experiment name must encode dataset + protocol + split/fold. Do not reuse UrbanSound8K experiment names for ESC-50 or Speech Commands.
