# Canonical paper configs

Prefer these entrypoints for new runs. Files are copies of the historical MAIN recipes so long-lived paths under `configs/kv260_*.json` remain valid.

| File | Model (paper name) | Protocol | Role |
|------|--------------------|----------|------|
| `student_ds_conv2d_h1_pyramid_sourcegroup.json` | DS-Conv2D-H1 Pyramid | `source_group_8_1_1` | MAIN no-teacher |
| `student_ds_conv2d_h1_pyramid_clean811.json` | DS-Conv2D-H1 Pyramid | `clean_8_1_1` | Strict-fold baseline |
| `student_ds_conv2d_h1_pyramid_clean811_mcisr.json` | DS-Conv2D-H1 Pyramid | `clean_8_1_1` + MC-ISR | Best clean811 method (v1) |
| `student_ds_res1d_se_fullclip.json` | DS-Res1D-SE | (see file) | Pure-Conv1d baseline |

`model_name` inside JSON may still use legacy keys (`kv260_audio_net_ds1d`, …); `train.py` accepts both paper and legacy names.
