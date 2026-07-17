# Architecture And FLOPs Analysis

## Why This Matters

The split experiment shows that the training pipeline can reach paper-like accuracy under random clip splitting. Separately, the architecture complexity still has a mismatch:

| Source | Params | Complexity |
|---|---:|---:|
| Paper Table 4 | 406K | 40M FLOPs |
| Current PyTorch implementation | 409.3K params | 230.2M MACs |

If one multiply-add is counted as two floating point operations, the current implementation is about 460.4M FLOPs.

## Paper Statements

The paper describes Table 4 as comparing model parameters and FLOPs. It reports the proposed single model as 406K parameters and 40M FLOPs.

Relevant extracted paper lines:

- Table 4 header defines Params and FLOPs: `Training-model-1DCNN.extracted.txt:887`.
- TCAM1DCNN row reports `406 K` and `40 M`: `Training-model-1DCNN.extracted.txt:898`.
- The text later says the FLOPs are "merely 40 M": `Training-model-1DCNN.extracted.txt:1069`.

The architecture in Table 2 uses input length 8000 and the following Conv1D output lengths:

```text
8000 -> 4000 -> 2000 -> 1000 -> 200 -> 40 -> 20
```

The current PyTorch implementation matches these output shapes.

## Current Count By Component

Counting Conv1D and Linear multiply-accumulates for one 8000-sample frame:

| Component | MACs | Params |
|---|---:|---:|
| Main Conv1D backbone + classifier | 144,017,920 | 235,722 |
| TAM time projection `F'''` | 606,720 | 454 |
| TAM `F_s` full Conv1D kernel 3 | 85,524,480 | 129,472 |
| CAM gate | 43,008 | 43,680 |
| Total current implementation | 230,192,128 | 409,328 |

The largest contributors are:

| Layer | MACs |
|---|---:|
| Conv2 | 65,536,000 |
| Conv3 | 36,864,000 |
| Conv4 | 24,576,000 |
| TAM1 `F_s` | 24,576,000 |
| TAM3 `F_s` | 24,576,000 |

## Variant Study

The table below tests whether reasonable interpretations of the ambiguous attention modules can explain the paper's 40M FLOPs.

| Variant | MACs | FLOPs if MAC=2 FLOPs | Params |
|---|---:|---:|---:|
| Current full implementation | 230.19M | 460.38M | 409.33K |
| Main backbone only, no TAM/CAM | 144.02M | 288.04M | 235.72K |
| Main + TAM projection + CAM, no `F_s` | 144.67M | 289.34M | 279.86K |
| Main + TAM projection + CAM + `F_s` 1x1 | 173.18M | 346.35M | 323.31K |
| Main + TAM projection + CAM + depthwise `F_s` k3 | 146.49M | 292.98M | 281.65K |
| Current `F_s` but CAM bottleneck 1 | 230.15M | 460.30M | 367.00K |

None of these reasonable variants reaches 40M at input length 8000.

To reach 40M MACs by scaling input length alone:

- Current full architecture would need input length about 1390 samples.
- Main-backbone-only architecture would need input length about 2222 samples.

Both contradict the Table 2 example input length of 8000.

## Stronger Hypothesis: 40M Is Params Times Batch Size

The paper says the training batch size was 100. Its reported complexity is numerically consistent with multiplying parameter count by batch size:

| Quantity | Value |
|---|---:|
| Paper reported params | 406K |
| Paper reported FLOPs | 40M |
| `40M / 406K` | 98.52 |
| Paper batch size | 100 |
| `406K * 100` | 40.6M |
| Current params without bias | 407,488 |
| `407,488 * 100` | 40,748,800 |
| Current params with bias | 409,328 |
| `409,328 * 100` | 40,932,800 |

This almost exactly explains the reported `40M`, but it is not standard FLOPs accounting. Standard Conv1D FLOPs must depend on input length, output length, input channels, output channels, and kernel size.

For example, just Conv2 has:

```text
input channels = 32
output channels = 32
output length = 4000
kernel = 16
MACs = 32 * 32 * 4000 * 16 = 65,536,000
```

So Conv2 alone already exceeds 40M MACs. This proves that the paper's 40M figure cannot be standard per-frame Conv1D MACs/FLOPs for the Table 2 model.

## What The Parameter Count Implies

The current implementation has 409.3K params with bias and about 407.5K without bias. That is close to the paper's 406K.

This matters because the parameter count supports the current high-parameter interpretation more than the lower-cost variants:

- Removing `F_s` or making it depthwise drops parameters to about 280K, far from 406K.
- CAM bottleneck 1 drops parameters to about 367K, also far from 406K.
- Current implementation is the closest match to 406K.

Therefore, the architecture parameter count and the reported FLOPs are internally hard to reconcile.

## Current Interpretation

The most defensible interpretation is:

```text
The current implementation is close to the paper in parameter count and Table 2 output shapes, but the paper's 40M FLOPs figure is not reproducible under standard Conv1D MAC/FLOP accounting for an 8000-sample input. The number is best explained as approximately params multiplied by the training batch size of 100.
```

This does not by itself explain the official fold accuracy gap, because the same architecture reaches about 93% under random clip split. It is a separate reproducibility issue: the paper's complexity accounting appears underspecified or inconsistent.

## Follow-Up Experiments

These should be run after the split-protocol question is documented:

1. Implement a complexity-ablation model family:
   - backbone only;
   - TAM only;
   - CAM only;
   - TCAM current;
   - TCAM with `F_s` 1x1;
   - TCAM with depthwise `F_s`;
   - TCAM with CAM bottleneck 1.

2. For each variant, report:
   - params with and without bias;
   - MACs and 2x FLOPs;
   - random split accuracy;
   - official fold accuracy.

3. Treat the paper's `40M FLOPs` number as an unverified claim until the authors' exact FLOPs tool or implementation is known.
