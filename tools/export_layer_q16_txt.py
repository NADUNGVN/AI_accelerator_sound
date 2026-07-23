#!/usr/bin/env python
"""Export checkpoint layers as per-layer Q16 text files.

This exporter is intended for RTL/HLS experiments that want one text file per
logical layer, for example:

    conv01_weight_q16.txt
    conv01_bias_q16.txt
    fc01_weight_q16.txt
    fc01_bias_q16.txt

The default mode folds BatchNorm into Conv layers before quantization. That
gives the hardware side a single affine convolution per Conv+BN block:

    y = conv(x, fused_weight) + fused_bias

The full PyTorch checkpoint remains the accuracy reference. This tool only
creates a hardware-facing weight-bank view of that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def load_numpy():
    try:
        import numpy as np  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NumPy is required for Q16 quantization. Run this in the server conda "
            "environment, for example: conda activate sound_env"
        ) from exc
    return np


def load_torch():
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required to read .pt checkpoints. Run this in the server conda "
            "environment, for example: conda activate sound_env"
        ) from exc
    return torch


@dataclass
class LayerExport:
    kind: str
    source_base: str
    weight: np.ndarray
    bias: np.ndarray
    source_names: list[str]
    transform: str
    note: str


def torch_load(path: Path) -> Any:
    torch = load_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state(path: Path) -> OrderedDict[str, torch.Tensor]:
    torch = load_torch()
    obj = torch_load(path)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "student", "weights"):
            value = obj.get(key)
            if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                return OrderedDict(value.items())
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return OrderedDict(obj.items())
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    np = load_numpy()
    return tensor.detach().cpu().numpy().astype(np.float32)


def sanitize_name(name: str) -> str:
    name = name.replace(".weight", "").replace(".bias", "")
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return name or "layer"


def state_has_bn(state: dict[str, torch.Tensor], prefix: str) -> bool:
    return all(
        f"{prefix}.{suffix}" in state
        for suffix in ("weight", "bias", "running_mean", "running_var")
    )


def bn_prefix_for_weight(weight_name: str, state: dict[str, torch.Tensor]) -> str | None:
    base = weight_name[: -len(".weight")]
    candidates: list[str] = []

    if base.endswith(".conv.conv"):
        candidates.append(base[: -len(".conv.conv")] + ".bn")
    if base.endswith(".conv"):
        candidates.append(base[: -len(".conv")] + ".bn")
    if base.endswith(".dw"):
        candidates.append(base + "_bn")
    if base.endswith(".pw"):
        candidates.append(base + "_bn")
    if base.endswith(".depthwise"):
        candidates.append(base + "_bn")
    if base.endswith(".pointwise"):
        candidates.append(base + "_bn")
    if base.endswith(".shortcut.0"):
        candidates.append(base[: -len(".0")] + ".1")

    for prefix in candidates:
        if state_has_bn(state, prefix):
            return prefix
    return None


def fold_conv_bn(
    conv_weight: np.ndarray,
    conv_bias: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    np = load_numpy()
    out_channels = conv_weight.shape[0]
    if conv_bias is None:
        conv_bias = np.zeros(out_channels, dtype=np.float32)
    inv_std = 1.0 / np.sqrt(running_var.astype(np.float64) + float(eps))
    alpha = bn_weight.astype(np.float64) * inv_std

    reshape = (out_channels,) + (1,) * (conv_weight.ndim - 1)
    fused_weight = conv_weight.astype(np.float64) * alpha.reshape(reshape)
    fused_bias = bn_bias.astype(np.float64) + (conv_bias.astype(np.float64) - running_mean.astype(np.float64)) * alpha
    return fused_weight.astype(np.float32), fused_bias.astype(np.float32)


def fold_linear_bn(
    linear_weight: np.ndarray,
    linear_bias: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    np = load_numpy()
    out_features = linear_weight.shape[0]
    if linear_bias is None:
        linear_bias = np.zeros(out_features, dtype=np.float32)
    inv_std = 1.0 / np.sqrt(running_var.astype(np.float64) + float(eps))
    alpha = bn_weight.astype(np.float64) * inv_std
    beta = bn_bias.astype(np.float64) - running_mean.astype(np.float64) * alpha

    fused_weight = linear_weight.astype(np.float64) * alpha.reshape(1, -1)
    fused_bias = linear_bias.astype(np.float64) + linear_weight.astype(np.float64) @ beta
    return fused_weight.astype(np.float32), fused_bias.astype(np.float32)


def maybe_fold_head_bn_into_linear(
    state: dict[str, torch.Tensor],
    base: str,
    weight: np.ndarray,
    bias: np.ndarray | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str, str]:
    np = load_numpy()
    # DS-Res1D-SE has head_bn immediately before fc. This fold is optional and
    # only applied when the dimensions prove the relationship.
    if base == "fc" and state_has_bn(state, "head_bn"):
        bn_weight = tensor_to_numpy(state["head_bn.weight"])
        if bn_weight.shape[0] == weight.shape[1]:
            folded_w, folded_b = fold_linear_bn(
                weight,
                bias,
                bn_weight,
                tensor_to_numpy(state["head_bn.bias"]),
                tensor_to_numpy(state["head_bn.running_mean"]),
                tensor_to_numpy(state["head_bn.running_var"]),
                eps,
            )
            return (
                folded_w,
                folded_b,
                [
                    f"{base}.weight",
                    f"{base}.bias",
                    "head_bn.weight",
                    "head_bn.bias",
                    "head_bn.running_mean",
                    "head_bn.running_var",
                ],
                "linear_head_bn_fused",
                "Linear layer with preceding head_bn folded into weight/bias.",
            )

    if bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
        source_names = [f"{base}.weight"]
        note = "Linear layer had no bias tensor; exported zero bias."
    else:
        source_names = [f"{base}.weight", f"{base}.bias"]
        note = "Raw Linear weight/bias."
    return weight, bias, source_names, "linear_raw", note


def collect_layers(
    state: OrderedDict[str, torch.Tensor],
    *,
    mode: str,
    bn_eps: float,
) -> tuple[list[LayerExport], list[str]]:
    torch = load_torch()
    np = load_numpy()
    layers: list[LayerExport] = []
    warnings: list[str] = []
    consumed_biases: set[str] = set()

    for weight_name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not weight_name.endswith(".weight"):
            continue

        weight = tensor_to_numpy(tensor)
        if weight.ndim not in {2, 3, 4}:
            continue

        base = weight_name[: -len(".weight")]
        bias_name = f"{base}.bias"
        raw_bias = tensor_to_numpy(state[bias_name]) if bias_name in state else None

        if weight.ndim in {3, 4}:
            if mode == "bn_fused":
                bn_prefix = bn_prefix_for_weight(weight_name, state)
                if bn_prefix:
                    folded_w, folded_b = fold_conv_bn(
                        weight,
                        raw_bias,
                        tensor_to_numpy(state[f"{bn_prefix}.weight"]),
                        tensor_to_numpy(state[f"{bn_prefix}.bias"]),
                        tensor_to_numpy(state[f"{bn_prefix}.running_mean"]),
                        tensor_to_numpy(state[f"{bn_prefix}.running_var"]),
                        bn_eps,
                    )
                    source_names = [
                        weight_name,
                        f"{bn_prefix}.weight",
                        f"{bn_prefix}.bias",
                        f"{bn_prefix}.running_mean",
                        f"{bn_prefix}.running_var",
                    ]
                    if raw_bias is not None:
                        source_names.append(bias_name)
                        consumed_biases.add(bias_name)
                    layers.append(
                        LayerExport(
                            kind="conv",
                            source_base=base,
                            weight=folded_w,
                            bias=folded_b,
                            source_names=source_names,
                            transform="conv_bn_fused",
                            note=f"Folded {weight_name} with {bn_prefix} using eps={bn_eps}.",
                        )
                    )
                else:
                    if raw_bias is None:
                        raw_bias = np.zeros(weight.shape[0], dtype=np.float32)
                        warnings.append(f"{weight_name}: no matching BN or bias; exported zero bias.")
                        source_names = [weight_name]
                    else:
                        consumed_biases.add(bias_name)
                        source_names = [weight_name, bias_name]
                    layers.append(
                        LayerExport(
                            kind="conv",
                            source_base=base,
                            weight=weight,
                            bias=raw_bias,
                            source_names=source_names,
                            transform="conv_raw_no_bn",
                            note="Conv layer exported without BN folding.",
                        )
                    )
            else:
                if raw_bias is None:
                    raw_bias = np.zeros(weight.shape[0], dtype=np.float32)
                    source_names = [weight_name]
                    note = "Raw Conv weight; no bias tensor, exported zero bias."
                else:
                    consumed_biases.add(bias_name)
                    source_names = [weight_name, bias_name]
                    note = "Raw Conv weight/bias."
                layers.append(
                    LayerExport(
                        kind="conv",
                        source_base=base,
                        weight=weight,
                        bias=raw_bias,
                        source_names=source_names,
                        transform="conv_raw",
                        note=note,
                    )
                )
        elif weight.ndim == 2:
            if mode == "bn_fused":
                linear_w, linear_b, source_names, transform, note = maybe_fold_head_bn_into_linear(
                    state,
                    base,
                    weight,
                    raw_bias,
                    bn_eps,
                )
                if raw_bias is not None:
                    consumed_biases.add(bias_name)
            else:
                if raw_bias is None:
                    linear_b = np.zeros(weight.shape[0], dtype=np.float32)
                    source_names = [weight_name]
                    note = "Raw Linear weight; no bias tensor, exported zero bias."
                else:
                    linear_b = raw_bias
                    source_names = [weight_name, bias_name]
                    consumed_biases.add(bias_name)
                    note = "Raw Linear weight/bias."
                linear_w = weight
                transform = "linear_raw"
            layers.append(
                LayerExport(
                    kind="fc",
                    source_base=base,
                    weight=linear_w,
                    bias=linear_b,
                    source_names=source_names,
                    transform=transform,
                    note=note,
                )
            )

    for name, tensor in state.items():
        if not name.endswith(".bias") or name in consumed_biases:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        # In bn_fused mode these standalone biases are usually BN beta values
        # already absorbed into a fused Conv. Keep a warning only for affine
        # modules that did not become a layer pair.
        if mode == "raw" and tensor.ndim == 1:
            continue
    return layers, warnings


def quantize_int16(arr: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    np = load_numpy()
    flat = arr.astype(np.float64).ravel(order="C")
    max_abs = float(np.max(np.abs(flat))) if flat.size else 1.0
    if max_abs < 1e-12:
        max_abs = 1.0
    if scale is None:
        scale = max_abs / 32767.0
    q = np.clip(np.rint(flat / scale), -32768, 32767).astype(np.int16)
    return q, float(scale)


def q_stats(arr: np.ndarray, q: np.ndarray, scale: float) -> dict[str, Any]:
    np = load_numpy()
    flat = arr.astype(np.float64).ravel(order="C")
    recon = q.astype(np.float64) * float(scale)
    err = recon - flat
    return {
        "shape": list(arr.shape),
        "numel": int(arr.size),
        "float_min": float(flat.min()) if flat.size else 0.0,
        "float_max": float(flat.max()) if flat.size else 0.0,
        "q_min": int(q.min()) if q.size else 0,
        "q_max": int(q.max()) if q.size else 0,
        "scale_float_equals_q_times_scale": float(scale),
        "max_abs_error": float(np.max(np.abs(err))) if err.size else 0.0,
    }


def format_q_values(values: np.ndarray, number_format: str) -> list[str]:
    if number_format == "decimal":
        return [str(int(v)) for v in values]
    if number_format == "hex":
        return [f"{int(v) & 0xFFFF:04X}" for v in values]
    raise ValueError(f"Unsupported number_format: {number_format}")


def write_q_file(path: Path, values: np.ndarray, number_format: str) -> None:
    lines = format_q_values(values, number_format)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    mode: str,
    name_style: str,
    number_format: str,
    bn_eps: float,
) -> dict[str, Any]:
    state = load_state(checkpoint)
    layers, warnings = collect_layers(state, mode=mode, bn_eps=bn_eps)
    if not layers:
        raise RuntimeError(f"No Conv/Linear layers found in checkpoint: {checkpoint}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counters = {"conv": 0, "fc": 0}
    manifest_layers: list[dict[str, Any]] = []
    total_words = 0

    for index, layer in enumerate(layers, start=1):
        counters[layer.kind] += 1
        if name_style == "sequential":
            export_name = f"{layer.kind}{counters[layer.kind]:02d}"
        elif name_style == "source":
            export_name = sanitize_name(layer.source_base)
        else:
            raise ValueError(f"Unsupported name_style: {name_style}")

        weight_q, weight_scale = quantize_int16(layer.weight)
        bias_q, bias_scale = quantize_int16(layer.bias)
        weight_file = f"{export_name}_weight_q16.txt"
        bias_file = f"{export_name}_bias_q16.txt"
        write_q_file(out_dir / weight_file, weight_q, number_format)
        write_q_file(out_dir / bias_file, bias_q, number_format)
        total_words += int(weight_q.size + bias_q.size)

        manifest_layers.append(
            {
                "index": index,
                "export_name": export_name,
                "kind": layer.kind,
                "source_base": layer.source_base,
                "source_names": layer.source_names,
                "transform": layer.transform,
                "note": layer.note,
                "files": {
                    "weight_q16_txt": weight_file,
                    "bias_q16_txt": bias_file,
                },
                "weight": q_stats(layer.weight, weight_q, weight_scale),
                "bias": q_stats(layer.bias, bias_q, bias_scale),
            }
        )

    manifest = {
        "format": "per_layer_q16_txt_v1",
        "source_checkpoint": str(checkpoint.resolve()),
        "mode": mode,
        "name_style": name_style,
        "number_format": number_format,
        "flatten_order": "pytorch_c_contiguous",
        "quantization": {
            "scheme": "symmetric_int16_per_tensor",
            "range": [-32768, 32767],
            "reconstruction": "float_value_approx = q16_value * scale_float_equals_q_times_scale",
            "scale_note": "Each exported weight file and bias file has its own scale.",
        },
        "batchnorm": {
            "folded_into_conv": mode == "bn_fused",
            "eps": bn_eps,
        },
        "counts": {
            "layers": len(layers),
            "conv_layers": counters["conv"],
            "fc_layers": counters["fc"],
            "txt_files": len(layers) * 2,
            "total_q16_words": total_words,
        },
        "warnings": warnings,
        "layers": manifest_layers,
    }
    manifest_path = out_dir / "manifest_q16.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Per-layer Q16 text export",
                "",
                f"Source checkpoint: `{checkpoint}`",
                f"Mode: `{mode}`",
                f"Name style: `{name_style}`",
                f"Number format: `{number_format}`",
                "",
                "Each layer has two text files: `*_weight_q16.txt` and `*_bias_q16.txt`.",
                "Values are one signed INT16 word per line unless `number_format=hex` was used.",
                "Use `manifest_q16.json` for tensor shapes, scales, source tensor names, and BN-fold details.",
                "",
                "Float reconstruction per tensor:",
                "",
                "```text",
                "float_value_approx = q16_value * scale_float_equals_q_times_scale",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def default_out_dir_for_model(model_dir: Path, mode: str, name_style: str, number_format: str) -> Path:
    return model_dir / f"layer_q16_txt_{mode}_{name_style}_{number_format}"


def export_deploy_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifests = []
    for model_dir in sorted(args.models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        checkpoint = model_dir / "model_full.pt"
        if not checkpoint.exists():
            continue
        out_dir = default_out_dir_for_model(model_dir, args.mode, args.name_style, args.number_format)
        manifest = export_checkpoint(
            checkpoint,
            out_dir,
            mode=args.mode,
            name_style=args.name_style,
            number_format=args.number_format,
            bn_eps=args.bn_eps,
        )
        manifests.append(
            {
                "model_dir": str(model_dir),
                "checkpoint": str(checkpoint),
                "out_dir": str(out_dir),
                "layers": manifest["counts"]["layers"],
                "txt_files": manifest["counts"]["txt_files"],
            }
        )
        print(f"[ok] {model_dir.name}: {manifest['counts']['txt_files']} txt files -> {out_dir}")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-layer Q16 .txt files from a PyTorch checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint.")
    parser.add_argument("--model_dir", type=Path, default=None, help="Folder containing model_full.pt.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--export_deploy_models",
        action="store_true",
        help="Export every deploy/student_models/*/model_full.pt folder.",
    )
    parser.add_argument(
        "--models_root",
        type=Path,
        default=REPO / "deploy" / "student_models",
        help="Root used with --export_deploy_models.",
    )
    parser.add_argument(
        "--mode",
        choices=("bn_fused", "raw"),
        default="bn_fused",
        help="bn_fused folds Conv+BN into affine Conv files; raw exports Conv/Linear tensors without folding.",
    )
    parser.add_argument(
        "--name_style",
        choices=("sequential", "source"),
        default="sequential",
        help="sequential gives conv01/fc01 names; source keeps sanitized PyTorch source names.",
    )
    parser.add_argument(
        "--number_format",
        choices=("decimal", "hex"),
        default="decimal",
        help="Write signed decimal integers or unsigned two's-complement hex words.",
    )
    parser.add_argument("--bn_eps", type=float, default=1e-5, help="BatchNorm epsilon used when folding BN.")
    args = parser.parse_args()

    if args.export_deploy_models:
        summary = export_deploy_models(args)
        summary_path = args.models_root / "LAYER_Q16_TXT_EXPORT_SUMMARY.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary -> {summary_path}")
        return

    checkpoint = args.checkpoint
    out_dir = args.out_dir
    if args.model_dir is not None:
        checkpoint = args.model_dir / "model_full.pt" if checkpoint is None else checkpoint
        out_dir = out_dir or default_out_dir_for_model(args.model_dir, args.mode, args.name_style, args.number_format)

    if checkpoint is None:
        raise SystemExit("Provide --checkpoint, --model_dir, or --export_deploy_models.")
    if out_dir is None:
        out_dir = checkpoint.parent / f"layer_q16_txt_{args.mode}_{args.name_style}_{args.number_format}"

    manifest = export_checkpoint(
        checkpoint,
        out_dir,
        mode=args.mode,
        name_style=args.name_style,
        number_format=args.number_format,
        bn_eps=args.bn_eps,
    )
    print(f"[ok] {manifest['counts']['txt_files']} txt files -> {out_dir}")


if __name__ == "__main__":
    main()
