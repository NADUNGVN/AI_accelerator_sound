# Paper premise pack

| Doc / asset | Content |
|---|---|
| [DATA.md](DATA.md) | Dataset structure, multi-dataset **seed status** |
| [NAMING.md](NAMING.md) | **SDP / OFP** paper names vs internal keys |
| [MODELS.md](MODELS.md) | Layer families, params, MACs/FLOPs + sources |
| [MODEL_B_KD.md](MODEL_B_KD.md) | **Model B 80%** = KD-protect (not AST); AST numbers separate |
| [BIAS_AND_CHECKPOINTS.md](BIAS_AND_CHECKPOINTS.md) | What bias is; full `.pt` + bias sidecar |
| [figures/](figures/) | Paper-style PNG/SVG (Abdoli/TCAM-inspired) |
| Root [README.md](../../README.md) | Academic overview |
| [STRUCTURE.md](../../STRUCTURE.md) | How to extend the repository |

Regenerate figures (preferred):

```bash
python tools/generate_paper_figures_v2.py
```

Export full `.pt` + bias package:

```bash
python tools/export_checkpoint_package.py --checkpoint <path.pt> --out_dir artifacts/checkpoints/<label> --label <label>
```
