#!/usr/bin/env python
"""Paper-quality figures for README §1.2 (pipeline) and §2 (two headline models).

Style references (Abdoli et al. arXiv:1904.08990; TCAM Xu et al.; project dataflow notebook):
  - framing / full-clip contrast on real waveform
  - vertical channel stacks with C@L annotations
  - inference aggregation diagram
  - stem-filter frequency magnitude from a real .pt when available

stdlib + numpy + matplotlib (py -3.11 env).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Try dataflow export for a real waveform
DATAFLOW = Path(
    r"D:\Research\Research_Development_System\5_Thesis_Optimization_Guides"
    r"\1_ai_accelerator_sound\dataflow_exports\urban8k_1dcnn_from_archive"
)

# Layer stack for DS-Conv2D-H1 Pyramid (verified with torch hooks, T=64000)
DS_LAYERS = [
    ("Input", "1 @ 64,000", "raw mono 4 s @ 16 kHz", "#e5e7eb"),
    ("Stem Conv2D-H1", "24 @ 16,000", "k=31, s=4, BN, ReLU", "#93c5fd"),
    ("DS Block 1", "32 @ 8,000", "DW k15 s2 → PW 1×1", "#60a5fa"),
    ("DS Block 2", "48 @ 4,000", "DW k15 s2 → PW 1×1", "#3b82f6"),
    ("DS Block 3", "64 @ 2,000", "DW k11 s2 → PW 1×1", "#2563eb"),
    ("DS Block 4", "96 @ 1,000", "DW k9 s2 → PW 1×1", "#1d4ed8"),
    ("DS Block 5", "128 @ 500", "DW k9 s2 → PW 1×1", "#1e40af"),
    ("DS Block 6", "160 @ 250", "DW k7 s2 → PW 1×1", "#1e3a8a"),
    ("DS Block 7", "160 @ 250", "DW k15 s1 → PW 1×1", "#172554"),
    ("Pyramid pool", "2,240", "Avg∥Max bins {1,2,4}", "#34d399"),
    ("Classifier", "10", "Dropout + Linear", "#fbbf24"),
]


def _save(fig, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    # also SVG for crisp paper use
    svg = path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT), "and", svg.name)
    return path


def load_waveform() -> tuple[np.ndarray, np.ndarray, str]:
    """Return t, y, caption from dataflow export or synthetic fallback."""
    try:
        import csv

        manifest = DATAFLOW / "selected_samples_manifest.csv"
        if manifest.exists():
            rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
            chosen = next(
                (r for r in rows if r.get("class") == "street_music"),
                rows[0],
            )
            preview = DATAFLOW / chosen["sample_dir"] / "v3_clip4s_tensor_1x64000_preview.csv"
            if preview.exists():
                t, y = [], []
                for row in csv.DictReader(preview.open(encoding="utf-8")):
                    if row.get("clip_tensor_C1_L64000") not in (None, ""):
                        t.append(float(row["time_s"]))
                        y.append(float(row["clip_tensor_C1_L64000"]))
                if len(t) > 100:
                    # decimate for plot
                    step = max(1, len(t) // 2500)
                    t_a = np.array(t[::step])
                    y_a = np.array(y[::step])
                    cap = f"UrbanSound8K · {chosen.get('class','')} · mono 16 kHz (dataflow export)"
                    return t_a, y_a, cap
    except Exception as exc:  # noqa: BLE001
        print("[warn] waveform fallback:", exc)

    t = np.linspace(0, 4, 4000)
    y = 0.15 * np.sin(2 * np.pi * 40 * t) * np.exp(-0.3 * t)
    y += 0.08 * np.sin(2 * np.pi * 220 * t) * (t > 1.2) * (t < 2.4)
    y += 0.03 * np.random.default_rng(0).normal(size=t.shape)
    return t, y, "Synthetic demo waveform (export unavailable)"


def fig_pipeline_framing():
    """Abdoli-style framing figure + full-clip contrast for our MAIN path."""
    t, y, cap = load_waveform()
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), gridspec_kw={"height_ratios": [1.15, 1.0]})

    # --- (a) literature multi-frame with overlap ---
    ax = axes[0]
    ax.plot(t, y, color="#1f77b4", lw=0.9)
    ax.set_xlim(0, 4)
    ymax = max(0.25, float(np.percentile(np.abs(y), 99)) * 1.3)
    ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("Amplitude")
    ax.set_title("(a) Literature-style framing (Abdoli / multi-frame 1D-CNN)", loc="left", fontsize=12, fontweight="bold")
    # frames 1.0-2.0 and 1.5-2.5 with 50% overlap
    for x0, x1, ls, label, yoff in [
        (1.0, 2.0, "-", "Frame s (1.0–2.0 s)", 0.92),
        (1.5, 2.5, "--", "Frame s+1 (1.5–2.5 s)", 0.78),
    ]:
        ax.axvline(x0, color="#111827", ls=ls, lw=1.4)
        ax.axvline(x1, color="#111827", ls=ls, lw=1.4)
        ax.annotate(
            "",
            xy=(x1, ymax * yoff),
            xytext=(x0, ymax * yoff),
            arrowprops=dict(arrowstyle="<->", color="#111827", lw=1.3),
        )
        ax.text((x0 + x1) / 2, ymax * (yoff + 0.05), label, ha="center", va="bottom", fontsize=9)
    ax.annotate(
        "",
        xy=(2.0, ymax * 0.55),
        xytext=(1.5, ymax * 0.55),
        arrowprops=dict(arrowstyle="<->", color="#b91c1c", lw=1.4),
    )
    ax.text(1.75, ymax * 0.58, "Overlap 50%", ha="center", color="#b91c1c", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.25)

    # --- (b) our full-clip ---
    ax = axes[1]
    ax.plot(t, y, color="#0f766e", lw=0.9)
    ax.set_xlim(0, 4)
    ax.set_ylim(-ymax, ymax)
    ax.axvspan(0, 4, color="#d1fae5", alpha=0.55, zorder=0)
    ax.annotate(
        "",
        xy=(4.0, ymax * 0.85),
        xytext=(0.0, ymax * 0.85),
        arrowprops=dict(arrowstyle="<->", color="#065f46", lw=1.6),
    )
    ax.text(
        2.0,
        ymax * 0.9,
        "Single full-clip frame · T = 64,000 samples (4.0 s @ 16 kHz)",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#065f46",
    )
    ax.set_title(
        "(b) This work — DS-Conv2D-H1 Pyramid (no multi-frame SUM)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.text(0.01, -0.18, cap, transform=ax.transAxes, fontsize=8, color="#4b5563")

    fig.suptitle("Audio input conditioning: multi-frame literature vs full-clip student", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig01_pipeline_framing_fullclip.png")


def fig_protocol_seed():
    """Clear protocol naming diagram."""
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Split protocols and seed status (paper naming)", fontsize=14, fontweight="bold", pad=12)

    cards = [
        (0.3, 3.2, 3.6, 3.4, "#ecfdf5", "#059669", "Source-Disjoint Protocol", "SDP 8-1-1",
         ["Internal key: source_group_8_1_1", "8 / 1 / 1 source (fsID) buckets", "train↔test fsID overlap = 0",
          "Seed 83 · fsid_classid_balanced_v1", "SEEDED · TRAINED (headline tracks)"]),
        (4.2, 3.2, 3.6, 3.4, "#eff6ff", "#2563eb", "Official-Fold Protocol", "OFP 8-1-1",
         ["Internal key: clean_8_1_1", "Train folds 3–10 · Val 2 · Test 1", "Official US8K fold partitions",
          "Seed 83", "SEEDED · TRAINED (OFP + MC-ISR)"]),
        (8.1, 3.2, 3.6, 3.4, "#fff7ed", "#ea580c", "Literature 9+1 Protocol", "L91",
         ["Internal key: paper_9_1", "9 train folds · no val · 1 test", "Optional literature only",
          "Not the deploy headline path", "SEEDED · optional runs"]),
    ]
    for x, y, w, h, face, edge, title, short, lines in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=2))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=11, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + h - 0.75, short, ha="center", va="top", fontsize=10, fontweight="bold", color="#111827")
        for i, line in enumerate(lines):
            ax.text(x + 0.18, y + h - 1.25 - i * 0.42, "• " + line, ha="left", va="top", fontsize=8.5, color="#1f2937")

    # multi-dataset strip
    ax.add_patch(FancyBboxPatch((0.3, 0.35), 11.4, 2.5, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="#fef2f2", edgecolor="#dc2626", linewidth=1.8))
    ax.text(6.0, 2.5, "Multi-dataset paper scope — seed status", ha="center", fontsize=12, fontweight="bold", color="#991b1b")
    ax.text(6.0, 1.95, "UrbanSound8K — primary · SEEDED and TRAINED", ha="center", fontsize=10)
    ax.text(6.0, 1.4, "ESC-50 — secondary comparison · loaders not implemented · NOT SEEDED · NOT TRAINED", ha="center", fontsize=10)
    ax.text(6.0, 0.85, "Speech Commands (subset) — edge / short-clip · loaders not implemented · NOT SEEDED · NOT TRAINED", ha="center", fontsize=10)

    _save(fig, "fig01b_protocol_seed_status.png")


def _draw_channel_stack(ax, x_center, y_bottom, height, n_visible, color, label_top, label_bottom=None):
    """Abdoli-style stacked channel cylinders (schematic)."""
    n = min(n_visible, 10)
    oval_h = height / (n + 1.5)
    w = 0.55
    for i in range(n):
        y = y_bottom + i * oval_h * 0.85
        ellipse = mpatches.Ellipse((x_center, y + oval_h * 0.35), w, oval_h * 0.7,
                                   facecolor=color, edgecolor="#1e3a5f", linewidth=0.8, alpha=0.92)
        ax.add_patch(ellipse)
    # dots in the middle if many channels
    if n_visible > n:
        ax.text(x_center, y_bottom + height * 0.45, "⋮", ha="center", va="center", fontsize=14, color="#1e3a5f")
    ax.text(x_center, y_bottom + height + 0.12, label_top, ha="center", va="bottom", fontsize=8, fontweight="bold")
    if label_bottom:
        ax.text(x_center, y_bottom - 0.18, label_bottom, ha="center", va="top", fontsize=7.5, color="#374151")


def fig_model_architecture(title: str, subtitle: str, filename: str, accent: str, recipe_box: list[str]):
    """Separate architecture plate for one headline model (shared topology, different recipe)."""
    fig = plt.figure(figsize=(13.5, 6.8))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.82])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.text(7, 6.55, subtitle, ha="center", fontsize=10, color="#374151")

    # waveform icon
    tw = np.linspace(0, 1, 80)
    yw = 0.25 * np.sin(2 * np.pi * 6 * tw)
    ax.plot(0.35 + tw * 0.7, 3.4 + yw, color=accent, lw=1.2)
    ax.text(0.7, 2.7, "waveform\n[B,1,64000]", ha="center", fontsize=8)

    stages = [
        (1.8, "1@64k", "Input", "#dbeafe", 8),
        (3.0, "24@16k", "Stem\nConv2D-H1", "#93c5fd", 9),
        (4.2, "32@8k", "DS1", "#60a5fa", 9),
        (5.3, "48@4k", "DS2", "#3b82f6", 9),
        (6.4, "64@2k", "DS3", "#2563eb", 8),
        (7.5, "96@1k", "DS4", "#1d4ed8", 8),
        (8.6, "128@500", "DS5", "#1e40af", 7),
        (9.7, "160@250", "DS6–7", "#1e3a8a", 7),
        (10.9, "2240", "Pyramid\nAvg∥Max", "#34d399", 6),
        (12.2, "10", "FC", "#fbbf24", 5),
    ]
    for x, shape, name, col, nvis in stages:
        _draw_channel_stack(ax, x, 2.3, 2.8, nvis, col, shape, name)
    # arrows
    for i in range(len(stages) - 1):
        x0 = stages[i][0] + 0.32
        x1 = stages[i + 1][0] - 0.32
        ax.annotate("", xy=(x1, 3.7), xytext=(x0, 3.7),
                    arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))

    ax.text(13.3, 3.7, "Class", ha="left", va="center", fontsize=10, fontweight="bold")

    # operator strip
    ax.add_patch(FancyBboxPatch((0.4, 0.35), 8.8, 1.5, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.2))
    ax.text(4.8, 1.55, "Layer family (shared topology)", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        4.8,
        0.95,
        "Conv2d kernel (1, k) · depthwise-separable · BN · ReLU · pyramid adaptive pool · Linear\n"
        "Params 101,674 · MACs/clip 61.85 M · FLOPs≈123.7 M (FLOPs = 2×MACs)",
        ha="center",
        fontsize=8.5,
        color="#1f2937",
    )

    # recipe box (differs between the two models)
    ax.add_patch(FancyBboxPatch((9.5, 0.35), 4.1, 1.5, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#fffbeb", edgecolor=accent, lw=1.6))
    ax.text(11.55, 1.55, "Training recipe (this model)", ha="center", fontsize=9, fontweight="bold", color=accent)
    for i, line in enumerate(recipe_box):
        ax.text(9.7, 1.2 - i * 0.28, line, ha="left", fontsize=8, color="#1f2937")

    _save(fig, filename)


def fig_inference_aggregation():
    """Contrast multi-frame aggregation (literature) vs single-pass full-clip (this work)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    # left: literature
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Literature multi-frame aggregation\n(Abdoli Fig. 4 style)", fontsize=11, fontweight="bold")
    # waveform bar
    ax.add_patch(Rectangle((0.5, 4.5), 9, 0.7, facecolor="#dbeafe", edgecolor="#1e40af"))
    ax.plot(np.linspace(0.6, 9.3, 200), 4.85 + 0.2 * np.sin(np.linspace(0, 30, 200)), color="#1d4ed8", lw=0.8)
    for i, lab in enumerate(["X₁", "X₂", "…", "Xₛ"]):
        ax.add_patch(Rectangle((0.7 + i * 1.1, 3.5), 0.9, 0.7, facecolor="#bfdbfe", edgecolor="#1e3a8a"))
        ax.text(1.15 + i * 1.1, 3.85, lab, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(5, 2.6), xytext=(2, 3.5), arrowprops=dict(arrowstyle="->", color="#111"))
    ax.annotate("", xy=(5, 2.6), xytext=(4.3, 3.5), arrowprops=dict(arrowstyle="->", color="#111"))
    ax.annotate("", xy=(5, 2.6), xytext=(5.5, 3.5), arrowprops=dict(arrowstyle="->", color="#111"))
    ax.add_patch(FancyBboxPatch((3.7, 1.7), 2.6, 0.9, boxstyle="round,pad=0.03", facecolor="#fde68a", edgecolor="#92400e"))
    ax.text(5, 2.15, "1D-CNN", ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(5, 1.1), xytext=(5, 1.7), arrowprops=dict(arrowstyle="->", color="#111"))
    ax.add_patch(FancyBboxPatch((3.5, 0.35), 3, 0.75, boxstyle="round,pad=0.03", facecolor="#fecaca", edgecolor="#991b1b"))
    ax.text(5, 0.72, "Aggregation → Decision", ha="center", va="center", fontsize=9)
    ax.text(5, 5.5, "Many short frames · SUM/vote · high MAC/clip", ha="center", fontsize=8, color="#6b7280")

    # right: this work
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("This work — full-clip single forward\n(DS-Conv2D-H1 Pyramid)", fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((0.5, 4.5), 9, 0.7, facecolor="#d1fae5", edgecolor="#065f46"))
    ax.plot(np.linspace(0.6, 9.3, 200), 4.85 + 0.2 * np.sin(np.linspace(0, 30, 200)), color="#047857", lw=0.8)
    ax.add_patch(Rectangle((1.5, 3.5), 7, 0.7, facecolor="#6ee7b7", edgecolor="#065f46"))
    ax.text(5, 3.85, "X = full clip  [1, 64 000]", ha="center", va="center", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(5, 2.6), xytext=(5, 3.5), arrowprops=dict(arrowstyle="->", color="#111", lw=1.5))
    ax.add_patch(FancyBboxPatch((2.8, 1.7), 4.4, 0.9, boxstyle="round,pad=0.03", facecolor="#a7f3d0", edgecolor="#065f46"))
    ax.text(5, 2.15, "DS-Conv2D-H1 Pyramid", ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(5, 1.1), xytext=(5, 1.7), arrowprops=dict(arrowstyle="->", color="#111", lw=1.5))
    ax.add_patch(FancyBboxPatch((3.5, 0.35), 3, 0.75, boxstyle="round,pad=0.03", facecolor="#fef08a", edgecolor="#a16207"))
    ax.text(5, 0.72, "Decision (10 logits)", ha="center", va="center", fontsize=9)
    ax.text(5, 5.5, "One forward · ~61.9 M MACs/clip · DPU-friendly", ha="center", fontsize=8, color="#6b7280")

    fig.suptitle("Inference path comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig02c_inference_aggregation.png")


def fig_stem_filter_response(ckpt: Path | None):
    """Frequency magnitude of stem Conv2D-H1 kernels (paper Fig.6 style)."""
    weights = None
    source = "random init (no checkpoint)"
    if ckpt and ckpt.exists():
        try:
            import torch
            import sys

            sys.path.insert(0, str(ROOT))
            from src.models import DSConv2DH1PyramidNet

            obj = torch.load(ckpt, map_location="cpu")
            state = obj
            if isinstance(obj, dict):
                for k in ("model_state_dict", "state_dict", "model"):
                    if k in obj:
                        state = obj[k]
                        break
            m = DSConv2DH1PyramidNet(
                num_classes=10, dropout=0.25, pool_type="pyramid_avgmax", pool_bins=[1, 2, 4], stem_type="single"
            )
            m.load_state_dict(state, strict=False)
            weights = m.stem.conv.weight.detach().cpu().numpy()  # [24,1,1,31]
            source = f"checkpoint: {ckpt.name}"
        except Exception as exc:  # noqa: BLE001
            print("[warn] stem filter load failed:", exc)

    if weights is None:
        rng = np.random.default_rng(0)
        weights = rng.normal(size=(24, 1, 1, 31)).astype(np.float32)

    n_show = 12
    idxs = np.linspace(0, weights.shape[0] - 1, n_show, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.5), sharex=True, sharey=True)
    sr = 16000.0
    for ax, idx in zip(axes.ravel(), idxs):
        h = weights[idx, 0, 0, :]
        # zero-pad FFT
        nfft = 512
        H = np.fft.rfft(h, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
        mag = np.abs(H)
        mag = mag / (mag.max() + 1e-8)
        ax.plot(freqs, mag, color="#2563eb", lw=1.1)
        ax.set_xlim(0, 8000)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"filter {idx}", fontsize=8)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("f (Hz)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mag", fontsize=8)
    fig.suptitle(
        f"Stem Conv2D-H1 frequency response (12 of 24 filters)\n{source}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig02d_stem_filter_frequency_response.png")


def fig_results_bars():
    """Results with paper protocol names."""
    rows = [
        ("SDP No-Teacher peak (f1)", 79.08, "#16a34a"),
        ("SDP Ensemble last-2", 79.89, "#15803d"),
        ("SDP KD-Student", 80.00, "#0f766e"),
        ("SDP MAIN refresh (3090)", 77.70, "#2563eb"),
        ("SDP Baseline H0 (3090)", 76.90, "#3b82f6"),
        ("OFP Baseline", 65.06, "#f59e0b"),
        ("OFP + MC-ISR", 67.70, "#d97706"),
        ("OFP + MC-ISR-v2 (rej.)", 64.95, "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    ax.barh(y, vals, color=colors, height=0.68, edgecolor="white")
    ax.axvspan(80, 85, color="#dcfce7", alpha=0.55, zorder=0)
    ax.axvline(80, color="#16a34a", ls="--", lw=1.2)
    ax.axvline(85, color="#15803d", ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Best-val test accuracy (%)")
    ax.set_xlim(0, 90)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_title("Current results under paper protocol names", fontsize=13, fontweight="bold")
    ax.text(82.5, -0.9, "Phase A band 80–85%", color="#166534", fontsize=8, ha="center")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig03_results_accuracy_bars.png")


def main():
    fig_pipeline_framing()
    fig_protocol_seed()
    fig_model_architecture(
        title="Model A — DS-Conv2D-H1 Pyramid · No-Teacher",
        subtitle="Source-Disjoint Protocol (SDP 8-1-1) · fold-1 peak · best-val test = 79.08% · params 101,674",
        filename="fig02a_model_noteacher_79p08.png",
        accent="#15803d",
        recipe_box=[
            "• No teacher / no KD",
            "• CE + class weights + mixup + EMA",
            "• SDP 8-1-1 · seed 83",
            "• Best-val checkpoint → test",
        ],
    )
    fig_model_architecture(
        title="Model B — DS-Conv2D-H1 Pyramid · KD-Student",
        subtitle="Source-Disjoint Protocol (SDP 8-1-1) · fold-1 · best-val test = 80.00% (ens 80.23%) · same topology",
        filename="fig02b_model_kd_student_80p00.png",
        accent="#0f766e",
        recipe_box=[
            "• AST teacher (train-time only)",
            "• KD-protect fine-tune recipe",
            "• Student deploy = this stack",
            "• Teacher weights never on board",
        ],
    )
    fig_inference_aggregation()
    ckpt = ROOT / "experiments/local_multifold_pyramid_base_f1_f3_50ep/fold_1/checkpoints/tcam_fold_1_best.pt"
    fig_stem_filter_response(ckpt if ckpt.exists() else None)
    fig_results_bars()

    # manifest
    manifest = {
        "figures": sorted(p.name for p in FIG.glob("fig0*.png")),
        "protocol_names": {
            "source_group_8_1_1": "Source-Disjoint Protocol (SDP 8-1-1)",
            "clean_8_1_1": "Official-Fold Protocol (OFP 8-1-1)",
        },
        "headline_models": {
            "A_noteacher": {"acc": 79.08, "file": "fig02a_model_noteacher_79p08.png"},
            "B_kd_student": {"acc": 80.00, "file": "fig02b_model_kd_student_80p00.png"},
        },
    }
    (FIG / "figure_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("OK", manifest)


if __name__ == "__main__":
    main()
