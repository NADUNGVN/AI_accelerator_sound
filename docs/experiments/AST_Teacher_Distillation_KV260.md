# AST Teacher Distillation to KV260 1D-CNN

## Goal

Use a fine-tuned AST model only as an offline teacher, then train the KV260-friendly
`KV260AudioNetDS1D` student with hard labels plus cached AST soft logits. The
student architecture remains the deployable Conv2D-H1 / depthwise-separable 1D-CNN.

## Implementation

- `tools/cache_ast_teacher_logits.py` exports AST logits into a fold-specific cache.
- `train.py` loads `distillation.teacher_logits_template` when present.
- `CachedUrbanSoundFrameDataset` attaches cached logits only to train frame records.
- `Trainer` adds KL-divergence KD loss to the existing supervised loss.
- Validation and test evaluation remain label-only and never consume teacher logits.

## Smoke Test Commands

```powershell
$PY = "C:\Users\Dawin\AppData\Local\Programs\Python\Python311\python.exe"

& $PY tools/cache_ast_teacher_logits.py `
  --fold 2 `
  --teacher_checkpoint_template "experiments/local_ast_teacher_finetune_f2_f3_12ep_deferred/fold_{fold}/checkpoints/best" `
  --output_template "experiments/smoke_ast_teacher_logits/fold_{fold}/teacher_logits.pt" `
  --splits train `
  --max_train_clips 30 `
  --max_val_clips 10 `
  --max_test_clips 10 `
  --batch_size 4 `
  --local_files_only `
  --overwrite

& $PY train.py `
  --fold 2 `
  --config configs/kv260_ds1d_pyramid_ast_teacher_kd_smoke.json `
  --exp_name smoke_kv260_ast_teacher_kd `
  --max_train_clips 30 `
  --max_val_clips 10 `
  --max_test_clips 10
```

## Smoke Result 2026-07-20

Fold 2 smoke was run with 30 train clips, 10 validation clips, and 10 test clips.

| Check | Result |
|---|---:|
| AST train-cache subset accuracy | 86.67% |
| Cached train-frame coverage | 30/30 |
| Student params | 101,674 |
| Student MACs/clip | 61,854,400 |
| Smoke val accuracy | 10.00% |
| Smoke best-val test accuracy | 10.00% |

The smoke accuracy is not a model-quality result because the subset is tiny and
training ran for only two epochs. It only verifies that cache creation, cache
lookup, KD loss, checkpointing, and evaluation all execute end to end.

## Fold 2 Result 2026-07-20

Teacher cache:

| Item | Value |
|---|---:|
| Train clips cached | 6,995 |
| AST teacher train accuracy | 92.44% |
| Cache path | `experiments/ast_teacher_logits_sourcegroup_train/fold_2/teacher_logits.pt` |

Student run:

| Metric | Baseline KV260 | AST-KD KV260 | Delta |
|---|---:|---:|---:|
| Best validation | 70.49% | 70.26% | -0.23 pp |
| Validation-selected test | 67.67% | 72.17% | +4.50 pp |
| Final test | 67.32% | 71.48% | +4.16 pp |
| Last-2 ensemble test | 68.82% | 71.71% | +2.89 pp |

Fold 2 is a positive signal for AST-KD, but not yet a thesis-level result. The
promotion decision needs the same comparison over all available folds.

## Full Fold Commands

Run this only after AST checkpoints exist for every requested fold.

```powershell
$PY = "C:\Users\Dawin\AppData\Local\Programs\Python\Python311\python.exe"

& $PY tools/cache_ast_teacher_logits.py `
  --folds 1-10 `
  --teacher_checkpoint_template "experiments/local_ast_teacher_full10_12ep/fold_{fold}/checkpoints/best" `
  --output_template "experiments/ast_teacher_logits_sourcegroup_train/fold_{fold}/teacher_logits.pt" `
  --splits train `
  --batch_size 8 `
  --local_files_only `
  --overwrite

& $PY tools/run_multifold.py `
  --config configs/kv260_ds1d_pyramid_ast_teacher_kd_val.json `
  --exp_name local_multifold_kv260_ast_teacher_kd_50ep `
  --folds 1-10 `
  --analyze `
  --eval_modes
```

## Promotion Rule

Promote AST-KD only if validation-selected test accuracy improves over the matching
non-KD KV260 baseline under the same `source_group_8_1_1` split. Do not use final
test accuracy as the selection criterion.
