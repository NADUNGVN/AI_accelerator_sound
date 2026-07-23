# Bias, weights, and `.pt` checkpoints

## What is a bias?

In a neural layer that computes

\[
y = Wx + b
\]

- \(W\) is the **weight** matrix (or convolution kernel),
- \(b\) is the **bias** vector (additive offset).

Bias lets the decision surface **not** be forced through the origin: a neuron can fire even when the weighted sum of inputs is small. Classic references: any standard ML textbook treatment of affine layers; in deep learning, BatchNorm’s affine transform \(y = \gamma \hat{x} + \beta\) uses \(\beta\) as a **shift bias** after normalization ([Ioffe & Szegedy, Batch Normalization, ICML 2015](https://arxiv.org/abs/1502.03167)).

### In **DS-Conv2D-H1 Pyramid** specifically

| Component | Bias behaviour |
|---|---|
| Depthwise / pointwise `Conv2d` | Created with **`bias=False`** (no per-filter additive \(b\) on the convolution itself) |
| `BatchNorm2d` | Learns **γ** (`.weight`) and **β** (`.bias`) — β is the BN bias |
| Final `Linear` classifier | Has **`fc.bias`** of size 10 (one offset per class) |

Approximate counts for the MAIN student (**101 674** total parameters):

| Group | ≈ count |
|---|---:|
| Weights (kernels, BN-γ, FC-W) | **100 400** |
| Biases (BN-β + FC-b) | **1 274** |

So “bias” here is **not** statistical estimation bias of the accuracy metric; it is the **learned additive parameters** of the network.

## Why keep `.pt` **and** a separate bias file?

| Artifact | Required? | Role |
|---|---|---|
| **Full `.pt`** (`*_full.pt` or training `tcam_fold_*_best.pt`) | **Yes — always** | Exact eval, resume, ensemble, KD student load |
| **`*_biases.pt`** | Export / deploy aid | Isolated BN-β + FC bias for DPU packing notes, audits, hardware buffer maps |
| **`*_weights.pt`** | Optional | Kernel-only package paired with biases |

Hardware toolchains often store **MAC weights** and **bias/scale** in different on-chip memories. Exporting bias separately makes that mapping explicit **without** replacing the full PyTorch checkpoint.

For custom RTL/HLS, there is also a per-layer Q16 text export:

```bash
python tools/export_layer_q16_txt.py --export_deploy_models
```

Its default `bn_fused` mode folds BatchNorm into each Conv layer and therefore
creates an effective `*_bias_q16.txt` file for every Conv+BN block, even though
the raw PyTorch Conv modules use `bias=False`. The generated
`manifest_q16.json` records the source tensors, shape, scale, and flatten order.

## How to export

```bash
python tools/export_checkpoint_package.py \
  --checkpoint experiments/local_multifold_pyramid_base_f1_f3_50ep/fold_1/checkpoints/tcam_fold_1_best.pt \
  --out_dir artifacts/checkpoints/sdp_noteacher_f1_79p08 \
  --label sdp_noteacher_f1_79p08

python tools/export_checkpoint_package.py \
  --checkpoint experiments/local_finetune_kdprotect_f1_20ep/fold_1/checkpoints/tcam_fold_1_best.pt \
  --out_dir artifacts/checkpoints/sdp_kd_student_f1_80p00 \
  --label sdp_kd_student_f1_80p00
```

Each package directory contains:

```text
*_full.pt                 # full state_dict  ← canonical
*_weights.pt
*_biases.pt               # bias sidecar
*_package_manifest.json   # per-tensor stats
README.md
```

## Paper wording (suggested)

> Checkpoints are stored as full PyTorch state dictionaries (`.pt`). For deployment analysis we additionally export bias tensors (BatchNorm \(\beta\) and classifier bias) as a separate file; convolution kernels in this architecture are bias-free and rely on BatchNorm affine parameters for per-channel shifts.
