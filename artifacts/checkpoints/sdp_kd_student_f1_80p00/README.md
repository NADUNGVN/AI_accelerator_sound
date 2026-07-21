# Checkpoint package: `sdp_kd_student_f1_80p00`

| File | Role |
|------|------|
| `sdp_kd_student_f1_80p00_full.pt` | **Full** state_dict (weights + biases). Use for eval / resume. |
| `sdp_kd_student_f1_80p00_weights.pt` | Convolution / Linear / BN-γ weights only |
| `sdp_kd_student_f1_80p00_biases.pt` | **Bias / BN-β** tensors only |
| `sdp_kd_student_f1_80p00_package_manifest.json` | Per-tensor bias stats + counts |

## What is bias?

A bias is the additive term \(b\) in \(y = Wx + b\). Without it, every hyperplane must pass through the origin in feature space. In this DS-Conv2D-H1 network:

- Depthwise/pointwise **Conv2d** are created with `bias=False`.
- **BatchNorm** still has learnable affine parameters: scale \(\gamma\) (`.weight`) and shift \(\beta\) (`.bias`).
- The **classifier Linear** has an explicit 10-dim `.fc.bias`.

Total parameters in the MAIN student: **101 674**, of which **~1 274** are bias tensors (BN-β + FC bias).

## Rule

Keep **`sdp_kd_student_f1_80p00_full.pt`** as the canonical artifact. Bias export does **not** replace the `.pt` checkpoint.
