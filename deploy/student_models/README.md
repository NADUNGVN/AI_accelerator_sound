# Student bundle — DS-Conv2D-H1 models for chip / DPU design

These are the **two best float PyTorch checkpoints** for the deployable student network.  
They are intended for **SoC / DPU / Vitis-AI work by students** — not for re-training from scratch.

| Model | Folder | Acc (SDP fold-1, best-val test) | Role |
|-------|--------|--------------------------------:|------|
| **A. No-Teacher** | `model_a_noteacher_79p08/` | **79.08%** | Main no-teacher recipe |
| **B. KD-Student** | `model_b_kd_student_80p00/` | **80.00%** (ens 80.23%) | Distilled student (teacher **not** included) |

**Architecture (both):** DS-Conv2D-H1 Pyramid · ~101.7k trainable params · ~61.9 M MACs/clip · input `[1, 64000]` @ 16 kHz mono · 10 classes.

---

## Do you need to train again for DPU?

**No — not for the default path.**

| Goal | Need retrain? | What to do |
|------|---------------|------------|
| Run / inspect weights offline | No | Load `*_full.pt` in PyTorch |
| Port topology to RTL / HLS / custom chip | No | Use layer table in `model_card.json` + code `src/models/ds_conv2d_h1_pyramid.py` |
| Deploy on **KV260 DPU** (Vitis-AI) | **Usually no retrain** | **Convert** float `.pt` → quantize (PTQ) → compile `.xmodel` |
| Accuracy collapses after INT8 quant | **Maybe** | Then consider QAT (quantization-aware training) — that *is* a new train loop |

### Recommended DPU flow (convert-only first)

```text
1. Float checkpoint  *_full.pt
2. Export ONNX / TorchScript (or Vitis-AI PyTorch parser)
3. Calibrate / quantize (PTQ) on a small subset of UrbanSound8K
4. Compile for DPU target (e.g. DPUCZDX8G on KV260)
5. Run on board; measure accuracy + FPS + power
```

Only if step 5 accuracy is unacceptable, open a **QAT** track (retrain with fake-quant). That is **Phase B refinement**, not a prerequisite to start chip design.

Teacher (AST) is **never** deployed. Model B already is the **student** weights after KD.

---

## Files in each model folder

| File | Use |
|------|-----|
| **`model_weights.h5`** | **Student file 1** — HDF5 weight bank (float32 tensors) |
| **`model_weights.mem`** | **Student file 2** — Vivado `$readmemh` hex (INT16) for BRAM/ROM |
| `export_h5_mem_manifest.json` | Address map + per-tensor INT16 scales for `.mem` |
| `model_full.pt` | Canonical float PyTorch (golden software reference) |
| `model_weights.pt` / `model_biases.pt` | Optional split packages |
| `model_card.json` | Shapes, params, protocol, metrics |
| `README.md` | Short pointer |

**Retrain?** Not required to create `.h5`/`.mem` — convert from `model_full.pt`.  
Details / compliance: [`docs/hardware/H5_MEM_REQUIREMENTS.md`](../../docs/hardware/H5_MEM_REQUIREMENTS.md).

```bash
python tools/export_h5_mem_for_fpga.py
```

Load example:

```python
import torch
from src.models import DSConv2DH1PyramidNet

ckpt = torch.load("deploy/student_models/model_a_noteacher_79p08/model_full.pt", map_location="cpu")
state = ckpt["model_state_dict"]
model = DSConv2DH1PyramidNet(
    num_classes=10, dropout=0.25, pool_type="pyramid_avgmax",
    pool_bins=[1, 2, 4], stem_type="single",
)
model.load_state_dict(state, strict=True)
model.eval()
```

---

## What students need for chip design

1. **Topology** — `src/models/ds_conv2d_h1_pyramid.py` (Conv2D height=1, DS blocks, pyramid pool).  
2. **Fixed shapes** — input `1×64000`, layer table in `model_card.json`.  
3. **Weights** — `model_full.pt` (or weights+biases pair).  
4. **Budget** — ~102k params, ~62 M MACs/clip (see paper docs).  
5. **Quant plan** — start with INT8 PTQ; report drop vs float 79%/80%.

Further reading: `docs/paper/BIAS_AND_CHECKPOINTS.md`, `docs/paper/MODELS.md`, `docs/hardware/`.
