# Protocol and run naming (paper display)

Internal config keys stay stable for scripts. **Paper / README prose** uses the display names below.

## Split protocols

| Internal key | Paper display name | Short form | Meaning |
|---|---|---|---|
| `source_group_8_1_1` | **Source-Disjoint Protocol** | **SDP 8-1-1** | Train/val/test built from **source (fsID) buckets** so no recording source appears in both train and test. Algorithm: `fsid_classid_balanced_v1`. Seed **83**. |
| `clean_8_1_1` | **Official-Fold Protocol** | **OFP 8-1-1** | Uses UrbanSound8K **official folds**: train folds 3–10, val fold 2, test fold 1 (when fold index = 1). No source-bucket packing. |
| `paper_9_1` | **Literature 9+1 Protocol** | **L91** | Nine official train folds, **no validation**, one test fold (optional literature comparison only). |

## Named runs (results tables)

| Colloquial / legacy | Paper display name |
|---|---|
| clean811 base | **OFP Baseline** (Official-Fold, standard CE+mixup+EMA recipe) |
| clean811 MC-ISR v1 | **OFP + MC-ISR** (machinery-cluster + source-robust loss) |
| clean811 MC-ISR v2 | **OFP + MC-ISR-v2** (**rejected**) |
| source_group MAIN / H0 | **SDP Baseline** or **SDP MAIN** |
| fold1 peak 79.08% | **SDP No-Teacher (fold-1 peak)** |
| KD 80.00% | **SDP KD-Student (fold-1)** |

## Models

| Internal / legacy | Paper name |
|---|---|
| `kv260_audio_net_ds1d` | **DS-Conv2D-H1 Pyramid** |
| no-teacher 79.08% run | **DS-Conv2D-H1 Pyramid — No-Teacher** |
| KD student 80.00% run | **DS-Conv2D-H1 Pyramid — KD-Student** |

Both headline accuracies share the **same layer stack**; they differ by **training recipe** (CE recipe vs teacher-protected distillation), not by topology.
