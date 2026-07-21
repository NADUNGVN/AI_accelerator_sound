# Split protocols

## Main path (deploy)

| | |
|---|---|
| Name | **`source_group_8_1_1`** |
| Algorithm | `fsid_classid_balanced_v1` |
| Structure | **8** train buckets / **1** val / **1** test (source-group) |
| Seed | **83** |
| Overlap source train↔test | **0** (required) |
| Checkpoint | **Best val** → test |
| Primary metric | `test_acc_best_val_model` |
| Config | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` |

## Optional literature

| Name | Train / Val / Test | Checkpoint | Primary metric |
|---|---|---|---|
| `paper_9_1` | 9 official folds / **none** / 1 | Last snapshot | `test_acc_last_snapshot` |
| `clean_8_1_1` | 8 / 1 / 1 **official folds** | Best val | best-val test |

## Do not use for headline

| Name | Why |
|---|---|
| `random_clip_9_1` | High source overlap → inflated accuracy |

## Comparability

- Numbers from **different protocols are not interchangeable**.  
- Always write protocol name next to accuracy.  
- H0 server baseline: source-safe fold1 best-val test **~76.90%**.  
