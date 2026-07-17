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
