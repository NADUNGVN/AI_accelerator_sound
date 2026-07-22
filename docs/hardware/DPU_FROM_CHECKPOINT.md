# From float `.pt` to DPU — retrain or convert?

## Short answer

**For the current best students (79.08% No-Teacher, 80.00% KD-Student): convert only.**  
You do **not** need to retrain before starting Vitis-AI / DPU / chip design work.

| Step | Action | Retrain? |
|------|--------|----------|
| 1 | Take `deploy/student_models/.../model_full.pt` | No |
| 2 | Export graph (ONNX / TorchScript / Vitis-AI parser) | No |
| 3 | INT8 **PTQ** calibration on a small US8K subset | No |
| 4 | Compile `.xmodel` for KV260 DPU | No |
| 5 | Board accuracy / FPS / power | — |
| 6 | Only if INT8 drop is too large | **Then** QAT (new training) |

Architecture is already **Conv2D height=1 + DS blocks** so it is closer to DPU operator support than multi-frame attention 1D-CNNs.

## What students should open first

```text
deploy/student_models/README.md
deploy/student_models/model_a_noteacher_79p08/model_full.pt
deploy/student_models/model_b_kd_student_80p00/model_full.pt
src/models/ds_conv2d_h1_pyramid.py
```

Teacher AST weights are **not** in the bundle (and must not go on the chip).
