# UrbanSound8K Research Strategy

## Goal

Build a 1D-CNN training path for UrbanSound8K that balances low parameter count with high clip-level accuracy, then reuse the same method for ESC-10.

The current target should be reported under two separate definitions:

| Target | Meaning | Current status |
|---|---|---|
| Paper reproduction | Match the reported TCAM1DCNN setup as closely as possible | Core setup matches, but the paper split is underspecified |
| Source-independent benchmark | Measure generalization without `fsID+classID` leakage | This is the fair target for thesis claims |
| Leakage control | Stratified random clip split | Useful only to show the model can learn and to explain paper-like accuracy |

## What Is Known

- Hardware is sufficient for full UrbanSound8K runs: RTX 3090 24GB, 32GB RAM, Ubuntu.
- UrbanSound8K data is valid: 8732 standard 10-class clips after excluding `rail_vehicle`.
- Audio decode is clean: no missing filtered files, no all-zero decoded clips, no NaN/Inf.
- The current TCAM1DCNN matches the paper's main Table 2 output shapes.
- Current parameter count is close to the paper: 409,328 params with bias versus reported 406K.
- Standard Conv1D complexity is not close to the paper's 40M FLOPs claim: current count is about 230.19M MACs per 8000-sample frame.
- The previous fold-1 result shows a split-dependent gap:
  - official predefined fold: about 55.9% ensemble accuracy;
  - random clip split: about 93.3% ensemble accuracy, with substantial source-label overlap.

## Split Protocols To Use

| Protocol | Use case | Leakage expectation |
|---|---|---|
| `paper_9_1` | Official UrbanSound8K predefined fold comparison | No `fsID+classID` overlap |
| `random_clip_9_1` | Paper-like random clip control | High `fsID+classID` overlap |
| `source_group_9_1` | Randomized but source-label independent control | No `fsID+classID` overlap |
| `clean_8_1_1` | Hyperparameter selection without test-fold peeking | No test-fold use for checkpoint selection |

The immediate research question is whether `source_group_9_1` behaves like `paper_9_1` or like `random_clip_9_1`.

## Immediate UrbanSound8K Runs

Before a full run, use these quick checks:

```bash
python train.py --fold 1 --config configs/source_group_msle.json --exp_name smoke_sourcegroup --epochs 1 --batch_size 8 --max_train_clips 30 --max_test_clips 20
```

Expected result: the command should finish in seconds to minutes, report `Source-label overlap ... 0`, save snapshots/metrics, and produce near-random accuracy. This checks runtime correctness, not model quality.

```bash
python train.py --fold 1 --config configs/source_group_msle.json --exp_name smoke_overfit_sourcegroup --epochs 20 --batch_size 8 --lr 0.001 --max_train_clips 10 --max_test_clips 10
```

Expected result: training accuracy should climb clearly above random chance. Test accuracy is not meaningful here because the subset is tiny and source-independent.

If those pass, run these on the RTX 3090 server:

```bash
python train.py --fold 1 --config configs/source_group_msle.json --exp_name sourcegroup_msle_fp32
python tools/analyze_experiment.py --exp_dir experiments/sourcegroup_msle_fp32/fold_1 --fold 1 --config configs/source_group_msle.json --eval_all_cycles --eval_modes --eval_train
```

Then run one more official fold:

```bash
python train.py --fold 2 --config configs/reproduce_msle.json --exp_name paper9_msle_fp32
python tools/analyze_experiment.py --exp_dir experiments/paper9_msle_fp32/fold_2 --fold 2 --config configs/reproduce_msle.json --eval_all_cycles --eval_modes --eval_train
```

Interpretation:

- If source-group accuracy is low like official folds, the >90% random result is mostly a leakage/split effect.
- If source-group accuracy is high, official fold 1 may be unusually hard and a full 10-fold run is required before changing architecture.
- If train accuracy remains near 99% while source-independent test accuracy stays low, the next work should focus on regularization, augmentation, and architecture generalization rather than capacity.

## Path Toward Low Params And >90%

Use a staged approach:

1. Establish the honest baseline on `paper_9_1` and `source_group_9_1`.
2. Add controlled regularization without changing the reported split: random gain, time shift, waveform masking, mixup, and nonzero-frame sampling.
3. Run architecture ablations: backbone only, TAM only, CAM only, TCAM, CTAM, depthwise `F_s`, `F_s` 1x1, CAM bottleneck variants.
4. Track params, MACs, official-fold accuracy, source-group accuracy, and random-clip accuracy for every run.
5. Only after UrbanSound8K has a defensible protocol, port the exact pipeline to ESC-10.

For thesis reporting, do not claim `>90%` without naming the split. A strong result should be phrased as `>90% under source-independent official/source-group evaluation`; a random clip result should be labeled as a leakage control.

## Proposed Improvement Track

The reproduction track is no longer the main optimization target. The proposed track can change loss, optimizer, augmentation, and architecture as long as each result reports params and MACs.

Two configs are available:

| Config | Purpose | Params/MAC profile |
|---|---|---|
| `configs/proposed_tcam_aug_ce.json` | Accuracy-first, keeps TCAM architecture but adds CE, AdamW, class weights, label smoothing, AMP, gradient clipping, and waveform augmentation | About 409K params, about 230M MAC/input frame, 15 frames/clip |
| `configs/proposed_efficient_fullclip.json` | Efficiency-first full-clip 1D-CNN with depthwise-separable residual blocks and SE | About 149K params, about 98.7M MAC/clip |

Run fast checks first:

```bash
python train.py --fold 1 --config configs/proposed_efficient_fullclip.json --exp_name smoke_efficient_fullclip --epochs 1 --batch_size 4 --max_train_clips 12 --max_test_clips 8
python train.py --fold 1 --config configs/proposed_tcam_aug_ce.json --exp_name smoke_tcam_aug_ce --epochs 1 --batch_size 8 --max_train_clips 20 --max_test_clips 10
```

Run 50-epoch comparisons before committing to long runs:

```bash
python train.py --fold 1 --config configs/proposed_tcam_aug_ce.json --exp_name proposed_tcam_aug_ce_50ep --epochs 50 --batch_size 100
python train.py --fold 1 --config configs/proposed_efficient_fullclip.json --exp_name proposed_efficient_fullclip_50ep --epochs 50 --batch_size 128
```

Analyze:

```bash
python tools/analyze_experiment.py --exp_dir experiments/proposed_tcam_aug_ce_50ep/fold_1 --fold 1 --config configs/proposed_tcam_aug_ce.json --eval_all_cycles --eval_modes --eval_train
python tools/analyze_experiment.py --exp_dir experiments/proposed_efficient_fullclip_50ep/fold_1 --fold 1 --config configs/proposed_efficient_fullclip.json --eval_all_cycles --eval_modes --eval_train
```
