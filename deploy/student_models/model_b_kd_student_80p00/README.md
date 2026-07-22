# Model B — KD-Student (80.00%)

Canonical weights: **`model_full.pt`** (also `model_weights.h5` / `model_weights.mem` for FPGA).

## Training (important)

| | |
|--|--|
| Recipe | **KD-protect** fine-tune (`local_finetune_kdprotect_f1_20ep`) |
| Teacher | **Same-family DS-Conv2D-H1** (`cycle_final` ~79%) — **not AST** |
| Student acc | **80.00%** best-val test · **80.23%** ensemble |
| On device | **This folder only** (student). No teacher files. |

AST (~90% research teacher; AST-KD student ~75%) is **not** this package.  
Details: [`docs/paper/MODEL_B_KD.md`](../../../docs/paper/MODEL_B_KD.md).

See parent [README.md](../README.md) for DPU convert path (no retrain by default).
