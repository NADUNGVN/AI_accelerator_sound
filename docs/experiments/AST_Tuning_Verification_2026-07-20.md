# AST Tuning Verification

Date: 2026-07-20

## Purpose

Verify whether lower learning rate and longer frozen-base warmup can improve AST
fine-tuning under the source-safe `source_group_8_1_1` protocol.

Fold 2 was selected as the first strict check because the previous AST run
overfit quickly and remained below fold 1:

```text
baseline fold 2:
best val       = 84.85%
best-val test  = 85.33%
final test     = 85.10%
best epoch     = 1
early stopped  = yes
```

All runs below used:

```text
model: MIT/ast-finetuned-audioset-10-10-0.4593
protocol: source_group_8_1_1
fold: 2
train/test source-label overlap: 0
train/val source-label overlap: 0
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

## Candidate A: Low Encoder LR, Freeze 6

Command:

```bash
python tools/finetune_ast_teacher.py \
  --exp_name ast_teacher_tuned_lowlr_freeze6_f2_20ep \
  --fold 2 \
  --epochs 20 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --accum_steps 4 \
  --encoder_lr 3e-6 \
  --head_lr 2e-4 \
  --freeze_base_epochs 6 \
  --early_stop_warmup 4 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --local_files_only
```

Result:

| Metric | Value |
|---|---:|
| Completed epochs | 7 / 20 |
| Early stopped | yes |
| Best epoch | 2 |
| Best val | 83.81% |
| Best-val test | 84.30% |
| Final val | 82.32% |
| Final test | 86.84% |

Decision:

```text
Reject for promotion. Final checkpoint test accuracy increased, but the
validation-selected checkpoint is worse than the previous baseline. Using the
final checkpoint would be test-set peeking unless the selection rule is changed
and validated across folds.
```

## Candidate B: Micro Encoder LR, Freeze 4

Command:

```bash
python tools/finetune_ast_teacher.py \
  --exp_name ast_teacher_tuned_microenc_freeze4_f2_16ep \
  --fold 2 \
  --epochs 16 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --accum_steps 4 \
  --encoder_lr 1e-6 \
  --head_lr 5e-4 \
  --freeze_base_epochs 4 \
  --early_stop_warmup 8 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --local_files_only
```

Result:

| Metric | Value |
|---|---:|
| Completed epochs | 8 / 16 |
| Early stopped | yes |
| Best epoch | 1 |
| Best val | 84.85% |
| Best-val test | 85.33% |
| Final val | 81.40% |
| Final test | 85.45% |

Decision:

```text
Reject for promotion. This exactly reproduces the previous best validation and
best-val test result, but does not improve it. After unfreezing, validation
falls while train accuracy rises above 99%, which confirms overfitting rather
than useful adaptation.
```

## Comparison

| Run | Best val | Best-val test | Final test | Promote? |
|---|---:|---:|---:|---|
| Previous AST baseline fold 2 | 84.85% | 85.33% | 85.10% | baseline only |
| Low LR freeze 6 | 83.81% | 84.30% | 86.84% | no |
| Micro LR freeze 4 | 84.85% | 85.33% | 85.45% | no |

## Finding

Lower encoder learning rate and longer freeze do not solve the fold-2
source-safe generalization problem. The best checkpoint remains an early
classifier-head checkpoint before meaningful encoder adaptation.

Persistent weak groups remain:

```text
engine_idling fsID 94632 -> 0/31, mostly air_conditioner
jackhammer fsID 180937 -> 35/79 in baseline, mostly engine_idling
drilling fsID 180937 -> 4/16 in baseline, mostly engine_idling/jackhammer
```

This means the current AST fine-tuning bottleneck is not simply "not enough
fine-tuning." It is fold/source-specific domain confusion among continuous
machine-noise classes.

## Recommendation

Do not promote these tuned AST configs to full 10-fold. The next useful AST
check is a head-only or representation-level protocol across all folds, for
example:

```text
freeze_base_epochs >= epochs
epochs: 2-4
select by validation
run folds 1-10
```

If the head-only AST teacher remains around 85-90% but fails the same source
groups, use AST as an upper-bound teacher for distillation into the KV260 1D-CNN
student rather than spending more runs on full encoder fine-tuning.
