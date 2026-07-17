# Reproduction Research Agenda

## Current Position

The implementation is not failing in the basic training pipeline. The same model and preprocessing produce two very different outcomes:

| Split protocol | Test clips | fsID+classID overlap | Last snapshot | Last-2 ensemble |
|---|---:|---:|---:|---:|
| Official UrbanSound8K fold 1 (`paper_9_1`) | 873 | 0 | 55.67% | 55.90% |
| Random clip split (`random_clip_9_1`) | 874 | 492 | 93.25% | 93.36% |

The paper reports 91.43% for the single model and 94.04% for the two-model snapshot ensemble. The result that matches the paper is the random clip split, not the official predefined fold split.

The paper text says only:

> For accuracy of evaluation, 10-fold or 5-fold cross-validation scheme were employed, respectively, and we took average of experiments as presented results.

This does not explicitly state that the official UrbanSound8K predefined folds were used.

## Code Audit Status

Checks that currently look correct:

- Dataset is filtered to the 10 paper classes: 8732 clips.
- `rail_vehicle` is excluded.
- Label order matches UrbanSound8K class IDs: `air_conditioner=0` through `street_music=9`.
- Audio loader is fail-fast: it does not substitute failed decode with silence.
- Audio is resampled to 16 kHz, mono, padded/truncated to 4 seconds.
- Frame length is 8000 samples and hop is 4000 samples, giving 15 frames per clip.
- Official `paper_9_1` protocol trains on 9 official folds and tests on 1 official fold.
- Evaluation uses SUM over softmax probabilities from all 15 frames.
- Model output shapes match Table 2 for the main Conv1D stages.
- The model can overfit: train clip accuracy reaches about 99.68% on official fold 1 training clips.

Issue fixed during this audit:

- The random split control previously sorted by full `path`. That is environment-dependent between Windows and Linux. It now uses a stable metadata key: `(label, fold, slice_file_name, fsID, classID)`. Existing random split artifacts remain valid as historical evidence, and analyzer keeps backward compatibility with the older `path_v1` split.

Remaining implementation ambiguities:

- Paper uses TensorFlow; repo uses PyTorch. `padding="same"` for even kernels may not be bit-identical across frameworks, though output shapes match.
- Paper reports 406K params and 40M FLOPs. Current implementation has 409,328 params with bias and about 230M Conv/Linear MACs when all Conv1D operations inside TAM/CAM are counted. This is a significant FLOPs accounting or implementation ambiguity.
- Paper does not fully specify the convolution `F_s` inside TAM. Current code uses `kernel_size=3`.
- Paper says the CAM gating uses two 1x1 convolution operations; the exact bottleneck width is ambiguous. Current code uses `channels // 2`.
- Paper says MSLE in TensorFlow. Current MSLE matches the usual `log1p(softmax)` vs `log1p(one_hot)` form, but TensorFlow/Keras clipping details may differ slightly.

## Dataset Measurements To Track Per Fold

The official predefined folds have no `fsID+classID` overlap between train and test:

| Test fold | Train clips | Test clips | fsID+classID overlap | fsID-only overlap |
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

Frame padding is not a hard bug, but it must be tracked:

- Total frame count after framing: 130,980.
- All-zero padded frames: 11,625.
- All-zero padded frame rate: 8.88%.

Class-dependent padding is important because short classes can inject many zero frames. It did not explain the official-vs-random split gap by itself, because nonzero-frame aggregation only raised official fold 1 from about 55.67% to about 56.01%.

## What Each Model Component Is Supposed To Do

### Basic 1D CNN Backbone

Conv1D learns filters directly over raw waveform samples. In this paper, the first filters are short waveform pattern detectors, and later filters operate on increasingly compressed temporal feature maps.

The backbone reduces temporal length like this:

| Stage | Operation | Output shape |
|---|---|---|
| Input | raw waveform frame | 8000 x 1 |
| Conv1 | kernel 32, stride 1, 32 channels | 8000 x 32 |
| Conv2 | kernel 16, stride 2, 32 channels | 4000 x 32 |
| Conv3 | kernel 9, stride 2, 64 channels | 2000 x 64 |
| Conv4 | kernel 6, stride 2, 64 channels | 1000 x 64 |
| Conv5 | kernel 3, stride 5, 128 channels | 200 x 128 |
| Conv6 | kernel 3, stride 5, 128 channels | 40 x 128 |
| Conv7 | kernel 3, stride 2, 256 channels | 20 x 256 |
| GAP | average over time | 256 |
| Softmax | class probability | 10 |

ReLU adds nonlinearity after each main convolution. Global average pooling reduces overfitting pressure compared with a large fully connected head, but it does not remove split-dependent generalization failure.

### TAM: Time Attention Module

TAM tries to learn which time positions in a feature map matter. It projects all channels to a single temporal weight vector with a 1x1 convolution and sigmoid, then uses that vector to gate a transformed version of the feature map. The residual output is:

```text
Y_TAM = Y + F_s(Y) * sigmoid(F'''(Y))
```

Expected benefit: suppress irrelevant time segments and emphasize useful sound events within the frame.

Implementation ambiguity: the paper does not clearly specify the kernel size/cost of `F_s`. Current code uses a 3-sample Conv1D.

### CAM: Channel Attention Module

CAM tries to learn which channels are useful. Since 1D convolution channels can act like learned frequency-band detectors, CAM first averages each channel over time, then uses 1x1 convolutions and sigmoid to create channel weights.

Expected benefit: emphasize channels that correspond to discriminative waveform patterns and suppress irrelevant channels.

Implementation ambiguity: current code uses a `channels // 2` bottleneck. The paper wording around channel count is not precise enough to prove this is identical.

### TCAM: Time Then Channel Attention

TCAM applies TAM first, then CAM:

```text
Y_TCAM = CAM(TAM(Y))
```

Expected benefit: first choose useful time positions, then recalibrate useful channels. The paper claims six stacked TCAM blocks are optimal before overfitting/complexity hurts.

### Snapshot Ensemble

The cosine learning-rate schedule creates four cycle checkpoints. The paper says later snapshots are better and that ensembling the last two models gives the best condition.

In our official fold 1 run, cycle choice did not solve the issue:

| Cycle | Official fold 1 test accuracy |
|---:|---:|
| 1 | 55.67% |
| 2 | 57.62% |
| 3 | 54.18% |
| 4 | 55.67% |

In random clip split, all cycles are paper-like:

| Cycle | Random clip test accuracy |
|---:|---:|
| 1 | 92.68% |
| 2 | 93.59% |
| 3 | 93.59% |
| 4 | 93.25% |

## Research Roadmap

### Phase 1: Prove Split Effect Across More Folds

Run at least official fold 2 before full 10-fold:

```bash
python train.py --fold 2 --config configs/rtx3090_config.json --exp_name paper9_msle_fp32
```

Analyze fold 2:

```bash
python tools/analyze_experiment.py --exp_dir experiments/paper9_msle_fp32/fold_2 --fold 2 --config configs/rtx3090_config.json --eval_all_cycles --eval_modes --eval_train
```

If fold 2 is also low while random split is high, the main reproduction fork is confirmed as split protocol/source leakage.

### Phase 2: Source-Group Split

Add a source-group split control where `fsID+classID` cannot overlap, even under random splitting. This will answer:

- Does paper-like accuracy require source-label overlap?
- How much of random split accuracy comes from source leakage?
- Is official fold 1 unusually hard, or is all source-independent evaluation hard?

### Phase 3: Architecture Ambiguity Ablations

Run controlled ablations after split behavior is understood:

- Plain 1D CNN without TAM/CAM.
- TAM only.
- CAM only.
- CTAM order: CAM then TAM.
- TCAM current implementation.
- CAM bottleneck variants: `1`, `channels // 4`, `channels // 2`.
- TAM `F_s` variants: kernel `1`, kernel `3`, depthwise Conv1D.
- TensorFlow-style same padding emulation for even kernels.

Each ablation should be run first on random split and one official fold. Random split tells whether the model can learn; official fold tells whether it generalizes.

### Phase 4: Loss And Numeric Controls

Compare:

- MSLE current implementation.
- TensorFlow/Keras-style MSLE with explicit clipping.
- CrossEntropy.
- FP32 vs AMP only after FP32 baseline is understood.

### Phase 5: Thesis-Quality Conclusion

The defensible statement is currently:

```text
The repository reproduces paper-level accuracy under random clip-level splitting, but not under official UrbanSound8K predefined fold 1. The paper states 10-fold CV but does not explicitly state official predefined UrbanSound8K folds. Because random split introduces source-label overlap and yields paper-like accuracy, split protocol is the primary suspected cause of the reproduction discrepancy.
```

This should be kept separate from a stronger claim that the paper is wrong. The current evidence supports "split protocol is underspecified and likely not official-fold/source-independent."
