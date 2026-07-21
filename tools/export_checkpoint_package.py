#!/usr/bin/env python
"""Export a deployable checkpoint package: full .pt + separate bias sidecar.

The full ``model.pt`` (or training ``tcam_fold_*_best.pt``) remains the source of
truth for resume/eval. The bias sidecar isolates affine offsets for:

* documentation (what is a bias in this network)
* FPGA/DPU packing (weights vs bias/BN affine often live in different buffers)
* audit of classifier bias and BatchNorm β

Usage
-----
python tools/export_checkpoint_package.py \\
  --checkpoint experiments/.../tcam_fold_1_best.pt \\
  --out_dir artifacts/checkpoints/noteacher_f1_79p08 \\
  --label noteacher_sds_f1_79p08
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train import build_model  # noqa: E402


def load_state(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "student"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # bare state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def split_bias_tensors(state: dict) -> tuple[dict, dict, list[dict]]:
    weights: dict[str, torch.Tensor] = {}
    biases: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    for name, tensor in state.items():
        is_bias = name.endswith(".bias") or name.endswith("bias")
        # BatchNorm affine: weight=γ, bias=β — keep β in bias package only
        if is_bias:
            biases[name] = tensor.detach().cpu().clone()
            rows.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "numel": int(tensor.numel()),
                    "kind": "bias",
                    "mean": float(tensor.float().mean()),
                    "std": float(tensor.float().std(unbiased=False)),
                    "min": float(tensor.float().min()),
                    "max": float(tensor.float().max()),
                }
            )
        else:
            weights[name] = tensor.detach().cpu().clone()
    return weights, biases, rows


def default_cfg() -> dict:
    return {
        "model_name": "ds_conv2d_h1_pyramid",
        "width_mult": 1.0,
        "dropout": 0.25,
        "pool_type": "pyramid_avgmax",
        "pool_bins": [1, 2, 4],
        "stem_type": "single",
        "extra_late_blocks": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export .pt + bias sidecar package")
    parser.add_argument("--checkpoint", required=True, help="Path to training .pt")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--label", default="model", help="Package label prefix")
    parser.add_argument("--config_json", default=None, help="Optional training config JSON")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = default_cfg()
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    state = load_state(ckpt)
    weights, biases, rows = split_bias_tensors(state)

    # Full checkpoint copy (always keep .pt)
    full_path = out / f"{args.label}_full.pt"
    torch.save(
        {
            "model_state_dict": state,
            "source_checkpoint": str(ckpt.resolve()),
            "config": cfg,
            "note": "Full state_dict including weights and biases. Prefer this for eval/resume.",
        },
        full_path,
    )

    # Weight-only + bias-only
    weight_path = out / f"{args.label}_weights.pt"
    bias_path = out / f"{args.label}_biases.pt"
    torch.save({"weights": weights, "config": cfg}, weight_path)
    torch.save({"biases": biases, "config": cfg}, bias_path)

    # Human-readable bias table
    manifest = {
        "label": args.label,
        "source_checkpoint": str(ckpt.resolve()),
        "params_total": int(sum(t.numel() for t in state.values())),
        "params_weights": int(sum(t.numel() for t in weights.values())),
        "params_biases": int(sum(t.numel() for t in biases.values())),
        "num_bias_tensors": len(biases),
        "bias_tensors": rows,
        "explanation": {
            "what_is_bias": (
                "In a linear map y = Wx + b, the vector b is the bias. "
                "It shifts the activation so a neuron need not rely only on zero-mean inputs. "
                "In this student, Conv2d layers use bias=False; affine offsets appear as "
                "BatchNorm β (.bn.bias) and the final Linear classifier .fc.bias."
            ),
            "why_separate_file": (
                "Deployment and audit often pack convolutions (weights) separately from "
                "BN scale/shift and classifier bias. The full .pt is still required for "
                "training resume and exact PyTorch eval; the bias file is an export aid."
            ),
        },
        "files": {
            "full_pt": full_path.name,
            "weights_pt": weight_path.name,
            "biases_pt": bias_path.name,
        },
    }
    with open(out / f"{args.label}_package_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme = out / "README.md"
    readme.write_text(
        f"""# Checkpoint package: `{args.label}`

| File | Role |
|------|------|
| `{full_path.name}` | **Full** state_dict (weights + biases). Use for eval / resume. |
| `{weight_path.name}` | Convolution / Linear / BN-γ weights only |
| `{bias_path.name}` | **Bias / BN-β** tensors only |
| `{args.label}_package_manifest.json` | Per-tensor bias stats + counts |

## What is bias?

A bias is the additive term \(b\) in \(y = Wx + b\). Without it, every hyperplane must pass through the origin in feature space. In this DS-Conv2D-H1 network:

- Depthwise/pointwise **Conv2d** are created with `bias=False`.
- **BatchNorm** still has learnable affine parameters: scale \(\\gamma\) (`.weight`) and shift \(\\beta\) (`.bias`).
- The **classifier Linear** has an explicit 10-dim `.fc.bias`.

Total parameters in the MAIN student: **101 674**, of which **~1 274** are bias tensors (BN-β + FC bias).

## Rule

Keep **`{full_path.name}`** as the canonical artifact. Bias export does **not** replace the `.pt` checkpoint.
""",
        encoding="utf-8",
    )

    print(f"[export] full     -> {full_path}")
    print(f"[export] weights  -> {weight_path}")
    print(f"[export] biases   -> {bias_path}")
    print(f"[export] bias tensors: {len(biases)}  params_bias={manifest['params_biases']}")


if __name__ == "__main__":
    main()
