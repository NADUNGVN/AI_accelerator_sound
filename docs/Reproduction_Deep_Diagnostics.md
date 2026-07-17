# Reproduction Deep Diagnostics

## Split Outcome
- official_fold1: protocol=paper_9_1, test=873, last=55.67%, ensemble=55.90%, fsID+classID overlap=None
- random_clip_fold1: protocol=random_clip_9_1, test=874, last=93.25%, ensemble=93.36%, fsID+classID overlap=492

## Official Fold Source Overlap

| Fold | Train | Test | fsID+classID overlap | fsID-only overlap |
|---:|---:|---:|---:|---:|
| 1 | 7859 | 873 | 0 | 4 |
| 2 | 7844 | 888 | 0 | 1 |
| 3 | 7807 | 925 | 0 | 0 |
| 4 | 7742 | 990 | 0 | 2 |
| 5 | 7796 | 936 | 0 | 0 |
| 6 | 7909 | 823 | 0 | 0 |
| 7 | 7894 | 838 | 0 | 1 |
| 8 | 7926 | 806 | 0 | 1 |
| 9 | 7916 | 816 | 0 | 1 |
| 10 | 7895 | 837 | 0 | 0 |

## Frame Padding
- All-zero padded frames: 11625/130980 (8.88%).

## Model
- Params with bias: 409328
- Params without bias: 407488
- Approx Conv/Linear MACs: 230192128
- Paper reported params/FLOPs: 406 K / 40 M

### Complexity Groups

| Group | MACs | Params with bias |
|---|---:|---:|
| main_backbone_fc | 144,017,920 | 235,722 |
| tam_time_projection | 606,720 | 454 |
| tam_fs_full_conv | 85,524,480 | 129,472 |
| cam_gate | 43,008 | 43,680 |

### Complexity Variants

| Variant | MACs | FLOPs if MAC=2 FLOPs | Params with bias |
|---|---:|---:|---:|
| current_full_count | 230.19M | 460.38M | 409.33K |
| main_backbone_only | 144.02M | 288.04M | 235.72K |
| main_plus_projection_cam_half_no_fs | 144.67M | 289.34M | 279.86K |
| main_plus_projection_cam_half_fs_k1 | 173.18M | 346.35M | 323.31K |
| main_plus_projection_cam_half_fs_depthwise_k3 | 146.49M | 292.98M | 281.65K |
| current_but_cam_bottleneck1 | 230.15M | 460.30M | 367.00K |

- To reach 40M MACs with the current architecture by scaling input length alone, the input would need to be about 1390.14 samples, not 8000.
- Even counting only the main backbone and classifier, the model is about 144.02M MACs.

| Layer | Expected shape | Found shape | Match |
|---|---|---|---|
| conv1 | [1, 32, 8000] | [1, 32, 8000] | True |
| conv2 | [1, 32, 4000] | [1, 32, 4000] | True |
| conv3 | [1, 64, 2000] | [1, 64, 2000] | True |
| conv4 | [1, 64, 1000] | [1, 64, 1000] | True |
| conv5 | [1, 128, 200] | [1, 128, 200] | True |
| conv6 | [1, 128, 40] | [1, 128, 40] | True |
| conv7 | [1, 256, 20] | [1, 256, 20] | True |

## Interpretation

- The implementation matches the main Table 2 Conv1D output shapes.
- Official predefined fold evaluation and random clip split evaluation behave very differently.
- Random clip split has source-label overlap and reaches paper-like accuracy; official fold 1 does not.
- This points to split protocol/source leakage as the primary reproduction fork, not a hardware or DataLoader issue.
