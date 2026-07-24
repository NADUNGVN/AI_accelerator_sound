# Canonical paper configs

Prefer these entrypoints for new runs. Files are copies of the historical MAIN recipes so long-lived paths under `configs/kv260_*.json` remain valid.

| File | Model (paper name) | Protocol | Role |
|------|--------------------|----------|------|
| `student_ds_conv2d_h1_pyramid_sourcegroup.json` | DS-Conv2D-H1 Pyramid | `source_group_8_1_1` | MAIN no-teacher |
| `student_ds_conv2d_h1_pyramid_clean811.json` | DS-Conv2D-H1 Pyramid | `clean_8_1_1` | Strict-fold baseline |
| `student_ds_conv2d_h1_pyramid_clean811_mcisr.json` | DS-Conv2D-H1 Pyramid | `clean_8_1_1` + MC-ISR | Best clean811 method (v1) |
| `student_ds_res1d_se_fullclip.json` | DS-Res1D-SE | (see file) | Pure-Conv1d baseline |
| `student_ds_conv2d_h1_pyramid_esc10_phase1.json` | DS-Conv2D-H1 Pyramid | `esc10_3_1_1_foldk_valnext_v1` | Phase 1 ESC-10 verification |
| `student_ds_conv2d_h1_pyramid_esc10_official4fold_phase1.json` | DS-Conv2D-H1 Pyramid | `esc10_official_4_1_cv` | ESC-10 literature-side 5-fold CV |
| `student_ds_conv2d_h1_pyramid_esc50_phase1.json` | DS-Conv2D-H1 Pyramid | `esc50_3_1_1_foldk_valnext_v1` | Phase 1 ESC-50 tuned light-augment/no-mixup |
| `student_ds_conv2d_h1_pyramid_speech_commands_phase1.json` | DS-Conv2D-H1 Pyramid | `speech_commands_v2_official12` | Phase 1 Speech Commands smoke/full-split |

Diagnostic ESC-50 configs are retained for audit:
`student_ds_conv2d_h1_pyramid_esc50_phase1_noaug_sanity.json` and
`student_ds_conv2d_h1_pyramid_esc50_phase1_lightaug_nomix.json`.

`model_name` inside JSON may still use legacy keys (`kv260_audio_net_ds1d`, …); `train.py` accepts both paper and legacy names.
