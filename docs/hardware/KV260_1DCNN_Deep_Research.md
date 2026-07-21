# KV260 1D-CNN Deep Research Plan

## Objective

The target is no longer strict paper reproduction. The target is a deployable 1D-CNN for environmental sound classification with:

- UrbanSound8K first, ESC-10 later;
- source-independent accuracy target: at least 90%;
- low parameter count and low inference cost;
- deployment path to AMD Kria KV260 / K26 SOM.

The current source-independent baseline is not good enough:

| Model/config | Split | Train acc | Test acc | Interpretation |
|---|---|---:|---:|---|
| TCAM + MSLE, 50 epochs | `source_group_9_1` | 90.72% | 68.62% | overfits source-independent split |
| random clip split | `random_clip_9_1` | high | about 93% | paper-like, but leakage control only |

Therefore, the main problem is generalization under source-independent evaluation, not basic learning capacity.

## KV260 Hardware Facts

Official AMD data for KV260/K26:

- Device: Zynq UltraScale+ MPSoC.
- Logic cells: 256K.
- DSP slices: about 1.2K.
- DDR memory: 4 GB.
- Typical KV260 DPU configuration: one B4096 DPU core in programmable logic.
- DPU peak: 1.23 TOPS INT8 at 300 MHz.

Important Vitis AI constraints:

- DPU acceleration depends on operator support and DPU configuration.
- Unsupported operators or unsupported layer patterns can be assigned to CPU.
- AMD recommends using the Vitis AI Model Inspector early.
- Layer ordering matters for compiler fusion. Prefer `Conv -> BatchNorm -> ReLU` style patterns.
- Zynq/Kria DPU inference is batch-size 1 in practice; batching does not improve single-core DPU efficiency.
- Vitis AI Optimizer channel pruning can reduce inference cost, often more than 2x, with fine-tuning.

## FLOPs/MACs Budget

Use MACs as the primary metric because convolution cost is naturally counted as multiply-accumulates. When a paper/tool reports FLOPs, explicitly state whether `1 MAC = 1 op` or `1 MAC = 2 FLOPs`.

For KV260, `1.23 TOPS INT8 peak` means the board can theoretically execute large models. But a good thesis model should not merely fit; it should leave headroom for pre/post-processing, streaming, thermal limits, and non-ideal DPU utilization.

Recommended per-clip inference targets:

| Tier | MACs per 4s clip | Params | Meaning |
|---|---:|---:|---|
| Excellent | <=100M MAC/clip | <=150K | strong KV260 target, low latency and low power |
| Good | 100M-300M MAC/clip | <=300K | acceptable if accuracy improves materially |
| Acceptable | 300M-1B MAC/clip | <=500K | only if needed to approach 90% |
| Too high for this goal | >1B MAC/clip | >500K | use only as teacher/baseline, not final edge model |

Current model cost estimates:

| Model | Params | MAC/input | Frames/clip | MAC/clip |
|---|---:|---:|---:|---:|
| TCAM1DCNN | 409,328 | 230.19M | 15 | about 3.45B |
| Efficient full-clip 1D-CNN, width 1.0 | 149,088 | 98.66M | 1 | about 98.66M |
| Efficient full-clip 1D-CNN, width 0.75 | 86,475 | 59.35M | 1 | about 59.35M |
| Efficient full-clip 1D-CNN, width 0.5 | 40,856 | 29.80M | 1 | about 29.80M |

Decision: the final model should target `<=100M MAC/clip` first. If accuracy cannot exceed 85%, allow a `<=300M MAC/clip` model. TCAM's 3.45B MAC/clip is useful as an accuracy-first teacher/baseline, but it is not the preferred KV260 final target.

## Deployment Implication: 1D-CNN But DPU-Safe

The research requirement is "must be 1D-CNN." For KV260 deployment, this should be implemented in a DPU-safe form:

```text
logical 1D convolution over time
implemented/exported as Conv2D with height=1 and kernel=(1, k)
input tensor shape: N x C x 1 x T
```

This preserves 1D-CNN semantics while improving the chance that Vitis AI compiles the graph to DPU. Pure PyTorch `Conv1d` might not be the safest deployment representation. The model should avoid unusual activations such as SiLU if Model Inspector/Compiler does not map them well; use ReLU/ReLU6 for the deployable graph.

Preferred deployable block:

```text
Conv2D(1 x k) -> BatchNorm -> ReLU
optional depthwise Conv2D(1 x k) -> pointwise Conv2D(1 x 1)
GlobalAveragePool -> Linear
```

Risk to check early:

- depthwise/group convolution support for the selected DPU configuration;
- adaptive pooling export pattern;
- max/avg dual pooling if not supported cleanly;
- SiLU/Swish activations;
- any residual/add pattern not fused or partitioned as expected.

## Dataset Findings That Drive Training

UrbanSound8K issues:

- Official/source-independent splits are much harder than random clip split.
- Random clip split can leak source-label information and overstate accuracy.
- Some classes contain many short clips. Paper-style zero-padding creates all-zero tail frames:
  - `gun_shot`: about 53% all-zero padded frames;
  - `car_horn`: about 36%;
  - `dog_bark`: about 19%.
- TCAM source-group run reached high train accuracy but stalled around 68-69% test, so the gap is source/domain robustness.

Therefore, training must optimize for source-independent validation during training, not after training.

## Training-Time Optimization Strategy

The next phase should optimize during training using a validation protocol, not train full runs blindly.

### 1. Use a source-independent train/val/test protocol

Add or use a clean source-group split:

```text
train: source groups not in val/test
val: source groups used for model selection and tuning
test: held out until final reporting
```

Do not use random clip validation for hyperparameter choice.

### 2. Architecture search constrained by MAC budget

Run width variants:

| Variant | Target |
|---|---|
| width 0.5 | prove minimum cost, about 30M MAC/clip |
| width 0.75 | strong low-cost target, about 59M MAC/clip |
| width 1.0 | current excellent-tier target, about 99M MAC/clip |
| width 1.25 | accuracy fallback, about 148M MAC/clip |

Stop expanding width once validation accuracy plateaus.

### 3. Train with deployment-aware operators

For final KV260 path, replace or export Conv1D as Conv2D-H1. Prefer:

- Conv/BN/ReLU;
- standard or depthwise-separable conv only after Model Inspector check;
- global average pooling only if compiler maps it cleanly;
- no SiLU in final deployable graph unless verified.

### 4. Use robust waveform augmentation

Use augmentation as part of training, not post-training:

- random gain;
- time shift;
- background noise at controlled SNR;
- waveform masking;
- polarity flip with low probability;
- optional mixup after baseline.

Keep augmentation moderate; if train accuracy stays low and validation does not improve, reduce it.

### 5. Add model selection and early stopping

Current `paper_9_1` and `source_group_9_1` have no validation-based model selection. For optimization, use a clean validation split and save best validation checkpoint. Report final test only once per selected model.

### 6. Quantization-aware training and pruning

Do not train a huge FP32 model and optimize only afterward. Once an architecture reaches reasonable validation accuracy:

1. Train FP32 baseline.
2. Fine-tune with quantization-aware training or Vitis-compatible fake quantization.
3. Apply channel pruning during/following training.
4. Fine-tune pruned model.
5. Re-measure source-independent validation accuracy.
6. Compile with Vitis AI Model Inspector/Compiler.

Use pruning only if accuracy is already promising. Pruning a weak model usually preserves weakness.

## Immediate Experiment Order

Run these in order:

1. Efficient full-clip 1D-CNN, width 1.0, 50 epochs.
2. TCAM augmented CE, 50 epochs, as accuracy-first comparison.
3. If efficient width 1.0 is below 75%, try width 1.25 or add validation-guided augmentation tuning.
4. If TCAM augmented improves far above 68%, distill its predictions into the efficient model.
5. Convert efficient model to DPU-safe Conv2D-H1 form and inspect with Vitis AI.

Decision thresholds:

| Result after 50 epochs | Next action |
|---|---|
| Efficient >=75% | continue tuning efficient model |
| Efficient >=85% | run 120-200 epochs and prepare QAT/pruning |
| TCAM augmented > efficient by >10 points | use TCAM as teacher, distill into efficient student |
| Both stay around 68-70% | change data strategy and validation protocol before longer runs |

## Thesis Claim Discipline

Use these categories in reporting:

- `random_clip_9_1`: leakage control, not final claim.
- `source_group_9_1`: source-independent research benchmark.
- `official UrbanSound8K folds`: official dataset benchmark.
- `KV260-ready`: only after Model Inspector/Compiler confirms DPU-compatible graph.

The final claim should be:

```text
1D-CNN, <=X MAC/clip, <=Y parameters, INT8-ready for KV260, Z% source-independent accuracy.
```

Do not claim "KV260 optimized" solely from low FLOPs; compile compatibility and INT8 validation must be shown.

## Source Notes

Official/reference sources used for this plan:

- AMD KV260 product page: device, logic cells, BRAM/URAM, DSP slices, DDR memory, and board-level specifications.
  - https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html
- AMD/Xilinx AI SDK UG1354 KV260 section: B4096 DPU, 1.23 TOPS INT8 peak at 300 MHz.
  - https://docs.amd.com/r/2.5-English/ug1354-xilinx-ai-sdk/KV260-Vision-AI-Starter-Kit
- Vitis AI UG1414 supported operators and DPU limitations: DPU support depends on operator, configuration, ISA, and compiler partitioning.
  - https://docs.amd.com/r/en-US/ug1414-vitis-ai/Supported-Operators-and-DPU-Limitations
- Vitis AI FAQ: Kria K26 support, common CNN layers, graph partitioning, and batch-size-1 implications for Zynq/Kria DPU.
  - https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/faq.html
- Vitis AI model development guidance: use Model Inspector early, layer ordering matters, unsupported subgraphs can go to CPU, pruning/channel pruning workflow.
  - https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html
