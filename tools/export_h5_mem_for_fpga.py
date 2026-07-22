#!/usr/bin/env python
"""Export student checkpoints to FPGA-oriented .h5 + .mem (no retrain).

Produces, for each model folder under deploy/student_models/:

* model_weights.h5  — HDF5 archive of all tensors (float32) + metadata
* model_weights.mem — Vivado $readmemh-compatible hex memory init
                      (default: INT16 Q1.14-style scale from max-abs per tensor,
                       concatenated in deterministic layer order)

Also writes export_manifest.json documenting layout so RTL can map addresses.

This does NOT create a Keras Sequential .h5 that load_model() can rehydrate
without a matching Keras graph — it is a **weight bank** for chip design.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def load_state(pt_path: Path) -> dict:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError(f"Cannot parse state_dict from {pt_path}")


def ordered_items(state: dict) -> list[tuple[str, np.ndarray]]:
    # Stable order: weights first by name, then biases, then buffers if any
    items = []
    for name in sorted(state.keys()):
        t = state[name]
        if not isinstance(t, torch.Tensor):
            continue
        arr = t.detach().cpu().numpy().astype(np.float32)
        items.append((name, arr))
    return items


def write_h5(path: Path, items: list[tuple[str, np.ndarray]], meta: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"layers": [], "total_params": 0}
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "ds_conv2d_h1_weight_bank_v1"
        f.attrs["framework_source"] = "pytorch"
        f.attrs["note"] = (
            "Weight bank for FPGA/chip design. Not a full Keras model graph. "
            "Load tensors by dataset path; see export_manifest.json for order."
        )
        for k, v in meta.items():
            if isinstance(v, (str, int, float)):
                f.attrs[k] = v
            else:
                f.attrs[k] = json.dumps(v)

        g = f.create_group("tensors")
        for name, arr in items:
            # HDF5 path-safe
            ds_name = name.replace(".", "/")
            parent = g
            parts = ds_name.split("/")
            for p in parts[:-1]:
                parent = parent.require_group(p)
            # Scalar tensors cannot use chunk/compression in h5py.
            if arr.ndim == 0 or arr.size <= 1:
                parent.create_dataset(parts[-1], data=np.atleast_1d(arr))
            else:
                parent.create_dataset(parts[-1], data=arr, compression="gzip")
            summary["layers"].append(
                {
                    "name": name,
                    "h5_path": f"tensors/{ds_name}",
                    "shape": list(arr.shape),
                    "numel": int(arr.size),
                    "dtype": "float32",
                    "min": float(arr.min()) if arr.size else 0.0,
                    "max": float(arr.max()) if arr.size else 0.0,
                }
            )
            summary["total_params"] += int(arr.size)
    return summary


def quantize_int16(arr: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Symmetric INT16 quantization. scale = max_abs / 32767."""
    flat = arr.astype(np.float64).ravel()
    max_abs = float(np.max(np.abs(flat))) if flat.size else 1.0
    if max_abs < 1e-12:
        max_abs = 1.0
    if scale is None:
        scale = max_abs / 32767.0
    q = np.clip(np.rint(flat / scale), -32768, 32767).astype(np.int16)
    return q, float(scale)


def write_mem(
    path: Path,
    items: list[tuple[str, np.ndarray]],
    *,
    bits: int = 16,
    layout: str = "per_tensor_int16",
) -> dict:
    """Write Vivado $readmemh hex file (one word per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if bits != 16:
        raise ValueError("Only 16-bit mem export implemented in v1")

    regions = []
    address = 0
    lines: list[str] = []
    # Header comments (Vivado ignores // comments in many flows; keep short)
    lines.append("// DS-Conv2D-H1 weight bank INT16 — Vivado $readmemh")
    lines.append("// Each line = one 16-bit word, hex, two's complement")
    lines.append(f"// layout={layout}")

    for name, arr in items:
        q, scale = quantize_int16(arr)
        start = address
        for word in q:
            # unsigned hex of two's complement bit pattern
            u = int(word) & 0xFFFF
            lines.append(f"{u:04X}")
            address += 1
        regions.append(
            {
                "name": name,
                "start_word": start,
                "end_word": address - 1,
                "n_words": int(q.size),
                "shape": list(arr.shape),
                "scale_float_equals_q_times_scale": scale,
                "quant": "symmetric_int16",
                "byte_order": "word_as_uint16_hex",
            }
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "bits": bits,
        "total_words": address,
        "total_bytes": address * (bits // 8),
        "regions": regions,
    }


def export_one(model_dir: Path, label: str) -> dict:
    pt = model_dir / "model_full.pt"
    if not pt.exists():
        raise FileNotFoundError(pt)
    card_path = model_dir / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}

    state = load_state(pt)
    items = ordered_items(state)

    meta = {
        "label": label,
        "paper_name": card.get("paper_name", label),
        "accuracy_best_val_test": card.get("accuracy_best_val_test", ""),
        "source_pt": str(pt.as_posix()),
    }

    h5_path = model_dir / "model_weights.h5"
    mem_path = model_dir / "model_weights.mem"
    h5_summary = write_h5(h5_path, items, meta)
    mem_summary = write_mem(mem_path, items)

    manifest = {
        "label": label,
        "retrain_required": False,
        "source": {
            "pytorch_full": "model_full.pt",
            "note": "Converted from float checkpoint; no retrain.",
        },
        "outputs": {
            "h5": "model_weights.h5",
            "mem": "model_weights.mem",
        },
        "h5": h5_summary,
        "mem": mem_summary,
        "compliance": {
            "h5_hdf5_weight_bank": True,
            "h5_keras_load_model_compatible": False,
            "mem_vivado_readmemh_hex": True,
            "mem_bitwidth": 16,
            "mem_quantization": "symmetric_int16_per_tensor_scale",
            "needs_rtl_address_map_agreement": True,
            "needs_hw_bitexact_verify": True,
        },
        "student_usage": {
            "h5": "Read float32 tensors (or re-quantize in your tool) via h5py / MATLAB hdf5read",
            "mem": "Initialize BRAM/ROM with $readmemh(\"model_weights.mem\", mem); map regions via this manifest",
        },
    }
    man_path = model_dir / "export_h5_mem_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] {label}: h5={h5_path.name} mem_words={mem_summary['total_words']} -> {man_path.name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_root",
        type=Path,
        default=REPO / "deploy" / "student_models",
    )
    args = parser.parse_args()
    roots = [
        args.models_root / "model_a_noteacher_79p08",
        args.models_root / "model_b_kd_student_80p00",
    ]
    all_m = []
    for r in roots:
        if r.is_dir():
            all_m.append(export_one(r, r.name))
    summary_path = args.models_root / "H5_MEM_EXPORT_SUMMARY.json"
    summary_path.write_text(json.dumps(all_m, indent=2), encoding="utf-8")
    print("summary ->", summary_path)


if __name__ == "__main__":
    main()
