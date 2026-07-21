# Model family card (paper-oriented)

Names encode **layer operator families**, not hardware product codes.

| Paper name | Class | `model_name` | Role | File |
|---|---|---|---|---|
| **DS-Conv2D-H1 Pyramid** | `DSConv2DH1PyramidNet` | `ds_conv2d_h1_pyramid` | Deployable **no-teacher / student** (MAIN) | `src/models/ds_conv2d_h1_pyramid.py` |
| **DS-Res1D-SE** | `DSRes1DSENet` | `ds_res1d_se` | Pure-Conv1d **no-teacher** baseline | `src/models/ds_res1d_se.py` |
| **AST Transformer Teacher** | `ASTTransformerTeacher` | HF id / tools | **Teacher only** (KD Track 3) | `src/models/ast_transformer_teacher.py` |
| TCAM-Attn1D | `TCAMAttn1DNet` | `tcam_attn1d` | Literature framed 1D-CNN baseline | `src/models/tcam_attn1d.py` |

Legacy identifiers (`kv260_audio_net_ds1d`, `efficient_audio_cnn1d`, `tcam1dcnn`, class aliases `KV260AudioNetDS1D`, …) remain valid for old configs and metrics JSON.

---

## 1. DS-Conv2D-H1 Pyramid (MAIN student)

### Layer character

Logical **1D temporal CNN** on mono waveform, **implemented as Conv2D with height=1** so every convolution is DPU/Vitis-AI friendly.

| Stage | Operators | Notes |
|---|---|---|
| Input | reshape `[B,1,T] → [B,1,1,T]` | `T=64000` (4 s @ 16 kHz) |
| Stem | `Conv2d(k=(1,31), s=(1,4))` + BN + ReLU | optional multi-scale stem |
| Body | Depthwise `Conv2d-H1` → Pointwise `1×1` (DS blocks) | progressive temporal downsample |
| Head | Pyramid adaptive **avg + max** over bins `{1,2,4}` | multi-scale temporal summary |
| Classifier | Dropout + Linear → 10 | clip-level logits |

### Complexity (MAIN config)

| Quantity | Value | Source |
|---|---:|---|
| Parameters (with bias) | **101 674** | `metrics.json` / `sum(p.numel())` |
| MACs / clip (Conv+Linear lower bound) | **61 854 400** | training metrics `model_conv_linear_macs_per_clip_eval` |
| FLOPs / clip (if 1 MAC = 2 FLOPs) | **≈123.7 M** | project convention in config `deployment_budget` |

---

## 2. DS-Res1D-SE (no-teacher software baseline)

### Layer character

Native **Conv1d** multi-scale stem + **depthwise-separable residual** blocks with **Squeeze-and-Excitation** (SiLU). Same full-clip input; **not** packed as Conv2D-H1.

| Stage | Operators |
|---|---|
| Stem | parallel Conv1d kernels `{9,31,63}`, stride 4 |
| Body | DSResBlock: DW → BN → SiLU → PW → BN → SiLU → SE → residual |
| Head | global avg ∥ max → BN → Dropout → Linear |

### Complexity

| Quantity | Value |
|---|---:|
| Parameters | **149 088** |
| MACs / clip | **≈98.7 M** (architecture design count) |
| FLOPs / clip (×2) | **≈197 M** |

---

## 3. AST Transformer Teacher (Track 3 only)

| Item | Value |
|---|---|
| Base | HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Layers | log-mel patches → patch embedding → **Transformer encoder (MHA)** → classifier |
| Role | fine-tune / cache logits → **distill** into DS-Conv2D-H1 student |
| Deploy | **never** — board runs student only |

Wrapper: `ASTTransformerTeacher` in `src/models/ast_transformer_teacher.py`. Training entry points remain under `tools/finetune_ast_teacher.py`, `tools/cache_ast_teacher_logits.py`.

---

## 4. How parameters and FLOPs / MACs are defined

We never mix three quantities:

1. **Parameters** — number of learned weights/biases. Independent of input length and batch size.  
2. **MACs per sample / clip** — multiply-accumulate ops for one forward. Depend on tensor shapes.  
3. **FLOPs** — floating-point ops; convention-dependent.

### Sources for counting conventions

| Source | What it states |
|---|---|
| [fvcore FLOP count](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py) | FLOP is not perfectly well-defined; common practice counts one fused multiply-add as **one FLOP** in their tool |
| [ptflops](https://github.com/sovrasov/flops-counter.pytorch) | Theoretical multiply-add operations and parameters |
| [TensorFlow profiler docs](https://www.tensorflow.org/api_docs/python/tf/compat/v1/profiler/ProfileOptionBuilder) | `float_operation` depends on registered op statistics |
| This repo | `docs/architecture/Architecture_FLOPs_Analysis.md` + `tools/flops_lower_bound_check.py` |

### Analytic lower bound (Conv)

For a convolution (1D or H=1 2D equivalent):

```text
MACs = L_out × C_out × (C_in / groups) × K
```

For a linear layer:

```text
MACs = F_in × F_out
```

This **excludes** bias adds, activations, residual adds, SE multiplies, pooling arithmetic — hence a **conservative lower bound**.

### Project FLOPs convention

Config field:

```json
"deployment_budget": {
  "flops_equivalent_convention": "FLOPs = 2 * MACs"
}
```

When a paper or table says “FLOPs” without definition, state the convention. Our **headline complexity** for the student is:

```text
params = 101,674
MACs/clip = 61.85 M
FLOPs/clip ≈ 123.7 M   (under FLOPs = 2 × MACs)
```

Under the fvcore “1 MAC = 1 FLOP” style, report **61.85 M FLOPs/clip** instead and label the convention.

---

## 5. Difference from “generic 1D-CNN” narratives

| Aspect | Framed TCAM-style 1D-CNN | DS-Res1D-SE | DS-Conv2D-H1 Pyramid |
|---|---|---|---|
| Input unit | short frame (e.g. 0.5 s) | full 4 s clip | full 4 s clip |
| Aggregation | SUM over many frames | one forward | one forward |
| Core ops | Conv1d + time/channel attention | Conv1d DS + SE | **Conv2d H=1** DS |
| Deploy packing | poor MAC/clip | software-friendly | **DPU-oriented** |
| Attention | TCAM | SE only | none (pyramid pool) |

Figures: `docs/paper/figures/fig02_model_architectures.svg`, `fig04_complexity_params_macs.svg`.
