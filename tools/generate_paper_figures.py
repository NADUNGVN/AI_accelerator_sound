"""Generate academic SVG figures for the main README (stdlib only)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "paper" / "figures"
RES = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def box(x, y, w, h, fill, stroke="#1f2937", sw=1.5, rx=6):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def text(x, y, s, size=13, weight="normal", anchor="middle", fill="#111827"):
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{s}</text>'
    )


def data_pipeline_svg():
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="520" viewBox="0 0 980 520">',
        '<rect width="980" height="520" fill="#fafafa"/>',
        text(490, 28, "UrbanSound8K data flow and seed protocols", 18, "bold"),
        box(30, 55, 200, 90, "#dbeafe"),
        text(130, 90, "UrbanSound8K", 14, "bold"),
        text(130, 112, "8,732 clips · 10 classes", 11),
        text(130, 130, "10 official folds · fsID", 11),
        box(280, 55, 200, 90, "#e0e7ff"),
        text(380, 90, "Preprocess", 14, "bold"),
        text(380, 112, "16 kHz mono · pad/crop 4 s", 11),
        text(380, 130, "T = 64,000 samples", 11),
        box(530, 55, 200, 90, "#d1fae5"),
        text(630, 90, "Model input", 14, "bold"),
        text(630, 112, "[B, 1, 64000] waveform", 11),
        text(630, 130, "full-clip (1 frame/clip)", 11),
        '<line x1="230" y1="100" x2="275" y2="100" stroke="#374151" stroke-width="1.8"/>',
        '<polygon points="275,96 285,100 275,104" fill="#374151"/>',
        '<line x1="480" y1="100" x2="525" y2="100" stroke="#374151" stroke-width="1.8"/>',
        '<polygon points="525,96 535,100 525,104" fill="#374151"/>',
        text(490, 185, "Split / seed status (reproducibility seed = 83 where applicable)", 14, "bold"),
        box(30, 210, 290, 140, "#ecfdf5", "#059669"),
        text(175, 240, "source_group_8_1_1  (MAIN)", 13, "bold", fill="#065f46"),
        text(175, 265, "fsid_classid_balanced_v1", 11),
        text(175, 285, "8 train / 1 val / 1 test source buckets", 11),
        text(175, 305, "train↔test fsID overlap = 0", 11),
        text(175, 325, "SEEDED · trained (Track 1–3)", 11, "bold", fill="#047857"),
        box(345, 210, 290, 140, "#eff6ff", "#2563eb"),
        text(490, 240, "clean_8_1_1  (strict folds)", 13, "bold", fill="#1e40af"),
        text(490, 265, "test fold1 · val fold2 · train 3–10", 11),
        text(490, 285, "official-fold protocol", 11),
        text(490, 305, "no source-group packing", 11),
        text(490, 325, "SEEDED · trained (MC-ISR)", 11, "bold", fill="#1d4ed8"),
        box(660, 210, 290, 140, "#fff7ed", "#ea580c"),
        text(805, 240, "paper_9_1  (literature)", 13, "bold", fill="#9a3412"),
        text(805, 265, "9 train folds / 0 val / 1 test", 11),
        text(805, 285, "optional comparison only", 11),
        text(805, 305, "not headline deploy path", 11),
        text(805, 325, "SEEDED · optional runs", 11, "bold", fill="#c2410c"),
        box(30, 375, 920, 115, "#fef2f2", "#dc2626"),
        text(490, 405, "Multi-dataset paper scope — seed status", 14, "bold", fill="#991b1b"),
        text(490, 430, "UrbanSound8K: SEEDED and trained (primary).", 12),
        text(
            490,
            452,
            "ESC-50: Phase 1 loader/config added · NOT FULL-TRAINED.",
            12,
        ),
        text(
            490,
            474,
            "Speech Commands (subset): Phase 1 loader/config added · NOT FULL-TRAINED.",
            12,
        ),
        "</svg>",
    ]
    write(FIG / "fig01_data_pipeline_and_seed.svg", "\n".join(parts))


def _stack(parts, x, y0, width, stages, line_color):
    for i, (lab, col) in enumerate(stages):
        yy = y0 + i * 70
        parts.append(box(x, yy, width, 52, col))
        parts.append(text(x + width / 2, yy + 32, lab, 12, "bold"))
        if i < len(stages) - 1:
            cx = x + width / 2
            parts.append(
                f'<line x1="{cx}" y1="{yy+52}" x2="{cx}" y2="{yy+68}" '
                f'stroke="{line_color}" stroke-width="2"/>'
            )


def model_comparison_svg():
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">',
        '<rect width="1100" height="620" fill="#fafafa"/>',
        text(
            550,
            28,
            "Two no-teacher approaches vs classic framed 1D-CNN / Transformer teacher",
            16,
            "bold",
        ),
        box(20, 50, 340, 540, "#f0fdf4", "#15803d", 2),
        text(190, 78, "A. DS-Conv2D-H1 Pyramid", 14, "bold", fill="#14532d"),
        text(190, 98, "(deployable student · MAIN)", 11, fill="#166534"),
        text(190, 125, "Input  [B,1,64000]  →  reshape [B,1,1,T]", 11),
    ]
    _stack(
        parts,
        50,
        145,
        280,
        [
            ("Stem Conv2D-H1 k31 s4", "#bbf7d0"),
            ("DSBlock DW→PW  ×6 (+ optional Res)", "#86efac"),
            ("Pyramid Avg+Max bins {1,2,4}", "#4ade80"),
            ("Dropout → Linear → 10 logits", "#22c55e"),
        ],
        "#166534",
    )
    parts += [
        text(190, 445, "Params: 101,674", 12, "bold"),
        text(190, 465, "MACs/clip: 61.85 M", 12),
        text(190, 485, "FLOPs (×2 MACs): ≈123.7 M", 12),
        text(190, 510, "Operators: Conv2d(H=1), BN2d,", 11),
        text(190, 528, "ReLU, AdaptivePool2d, Linear", 11),
        text(190, 555, "DPU / Vitis-AI friendly", 12, "bold", fill="#14532d"),
        box(380, 50, 340, 540, "#eff6ff", "#1d4ed8", 2),
        text(550, 78, "B. DS-Res1D-SE", 14, "bold", fill="#1e3a8a"),
        text(550, 98, "(pure Conv1d no-teacher)", 11, fill="#1e40af"),
        text(550, 125, "Input  [B,1,64000]  native Conv1d", 11),
    ]
    _stack(
        parts,
        410,
        145,
        280,
        [
            ("Multi-scale stem k={9,31,63} s4", "#bfdbfe"),
            ("DSResBlock DW+PW+SE+Res ×7", "#93c5fd"),
            ("Global Avg ∥ Max pool → 320-d", "#60a5fa"),
            ("BN → Dropout → Linear → 10", "#3b82f6"),
        ],
        "#1d4ed8",
    )
    parts += [
        text(550, 445, "Params: 149,088", 12, "bold"),
        text(550, 465, "MACs/clip: ≈98.7 M", 12),
        text(550, 485, "FLOPs (×2 MACs): ≈197 M", 12),
        text(550, 510, "Operators: Conv1d, BN1d, SiLU,", 11),
        text(550, 528, "SE, Dropout1d, Linear", 11),
        text(550, 555, "Software baseline (not DPU-packed)", 12, "bold", fill="#1e3a8a"),
        box(740, 50, 340, 250, "#fef3c7", "#b45309", 2),
        text(910, 78, "Classic framed 1D-CNN", 13, "bold", fill="#92400e"),
        text(910, 98, "TCAM-Attn1D (Xu et al.)", 11),
        text(910, 125, "Frame 0.5 s (T=8000) × 15", 11),
        text(910, 145, "Conv1d + TAM + CAM each stage", 11),
        text(910, 165, "Params ≈ 410 k", 11),
        text(910, 185, "MACs ≈ 230 M / frame", 11),
        text(910, 205, "≈ 3.45 B MACs / clip (×15)", 11, "bold"),
        text(910, 230, "Not the deployable main path", 11, "bold", fill="#9a3412"),
        text(910, 255, "(attention 1D-CNN literature baseline)", 10),
        box(740, 320, 340, 270, "#fae8ff", "#7e22ce", 2),
        text(910, 348, "Teacher: AST Transformer", 13, "bold", fill="#6b21a8"),
        text(910, 370, "(Track 3 KD only)", 11),
        text(910, 398, "log-mel patches → patch embed", 11),
        text(910, 418, "Transformer encoder (MHA)", 11),
        text(910, 438, "AudioSet-pretrained HF AST", 11),
        text(910, 458, "Fine-tune ~90%+ train/cache", 11),
        text(910, 485, "NOT deployed on KV260", 12, "bold", fill="#6b21a8"),
        text(910, 510, "Distills into DS-Conv2D-H1 student", 11),
        text(910, 545, "Layer family ≠ 1D-CNN", 11, "bold"),
        "</svg>",
    ]
    write(FIG / "fig02_model_architectures.svg", "\n".join(parts))


def results_bar_svg():
    rows = [
        ("Single peak (src-group f1)", 79.08, "#16a34a"),
        ("Ensemble last-2", 79.89, "#15803d"),
        ("KD student single", 80.00, "#0f766e"),
        ("3090 MAIN refresh", 77.70, "#2563eb"),
        ("3090 H0 baseline", 76.90, "#3b82f6"),
        ("clean811 base", 65.06, "#f59e0b"),
        ("clean811 MC-ISR v1", 67.70, "#d97706"),
        ("clean811 MC-ISR v2", 64.95, "#dc2626"),
    ]
    w, h = 920, 480
    left, right, top, bottom = 280, 60, 60, 50
    chart_w = w - left - right
    chart_h = h - top - bottom
    xmax = 90.0
    x80 = left + chart_w * (80 / xmax)
    x85 = left + chart_w * (85 / xmax)
    n = len(rows)
    bar_h = chart_h / n * 0.62
    gap = chart_h / n
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#fafafa"/>',
        text(w / 2, 32, "Current results — best-val test accuracy (%)", 16, "bold"),
        f'<rect x="{x80}" y="{top}" width="{x85 - x80}" height="{chart_h}" fill="#dcfce7" opacity="0.7"/>',
        f'<line x1="{x80}" y1="{top}" x2="{x80}" y2="{top + chart_h}" stroke="#16a34a" stroke-dasharray="4,3" stroke-width="1.5"/>',
        f'<line x1="{x85}" y1="{top}" x2="{x85}" y2="{top + chart_h}" stroke="#15803d" stroke-dasharray="4,3" stroke-width="1.5"/>',
        text(x80, top - 8, "80%", 10, fill="#166534"),
        text(x85, top - 8, "85%", 10, fill="#14532d"),
    ]
    for i, (lab, val, col) in enumerate(rows):
        cy = top + gap * i + gap * 0.2
        bw = chart_w * (val / xmax)
        out.append(box(left, cy, bw, bar_h, col, col, 0, 3))
        out.append(text(left - 10, cy + bar_h * 0.7, lab, 11, anchor="end"))
        out.append(text(left + bw + 8, cy + bar_h * 0.7, f"{val:.2f}%", 11, anchor="start"))
    out.append(
        text(
            w / 2,
            h - 16,
            "Green band = Phase A target 80–85%. MC-ISR v2 rejected. Peak 79.08% is fold-1 high-water mark.",
            10,
            fill="#4b5563",
        )
    )
    out.append("</svg>")
    body = "\n".join(out)
    write(FIG / "fig03_results_accuracy_bars.svg", body)
    write(RES / "results_accuracy_bars.svg", body)


def complexity_svg():
    rows = [
        ("DS-Conv2D-H1 Pyramid", 101.7, 61.9, "#16a34a"),
        ("DS-Res1D-SE", 149.1, 98.7, "#2563eb"),
        ("TCAM / frame", 410.2, 230.2, "#d97706"),
    ]
    w, h = 900, 420
    max_mac = 400.0
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#fafafa"/>',
        text(
            w / 2,
            30,
            "Complexity comparison: parameters (k) and MACs (M) per inference unit",
            15,
            "bold",
        ),
        text(
            w / 2,
            52,
            "Convention: MAC lower-bound for Conv/Linear; FLOPs ≈ 2×MACs (see docs/paper/MODELS.md)",
            11,
            fill="#4b5563",
        ),
    ]
    y0 = 90
    for i, (lab, pk, mac, col) in enumerate(rows):
        y = y0 + i * 90
        parts.append(text(20, y + 18, lab, 12, "bold", anchor="start"))
        parts.append(text(20, y + 38, f"params {pk:.1f}k", 11, anchor="start", fill="#374151"))
        bw = min(600.0, 600.0 * (mac / max_mac))
        parts.append(box(220, y, bw, 28, col, col, 0, 4))
        parts.append(text(230 + bw, y + 19, f"{mac:.1f} M MACs", 11, anchor="start"))
    parts.append(
        text(
            20,
            370,
            "TCAM multi-frame SUM ≈ 230.2 M × 15 ≈ 3.45 B MACs/clip — outside KV260-class budget.",
            12,
            "bold",
            fill="#991b1b",
            anchor="start",
        )
    )
    parts.append(
        text(
            20,
            395,
            "Student MAIN (DS-Conv2D-H1) stays at 61.9 M MACs/clip and 101.7k params.",
            12,
            anchor="start",
            fill="#14532d",
        )
    )
    parts.append("</svg>")
    write(FIG / "fig04_complexity_params_macs.svg", "\n".join(parts))


def tracks_gap_svg():
    tracks = [
        ("Track 1 Single", 79.08),
        ("Track 2 Ensemble", 79.89),
        ("Track 3 KD student", 80.00),
    ]
    w, h = 860, 340
    left = 180
    chart_w = 600

    def xv(v):
        return left + chart_w * ((v - 70) / 20)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#fafafa"/>',
        text(w / 2, 28, "Phase A progress toward 80–85% targets", 16, "bold"),
    ]
    for i, (lab, cur) in enumerate(tracks):
        y = 70 + i * 80
        parts.append(text(20, y + 22, lab, 12, "bold", anchor="start"))
        parts.append(box(xv(70), y, chart_w, 24, "#e5e7eb", "#e5e7eb", 0, 4))
        parts.append(box(xv(80), y, xv(85) - xv(80), 24, "#bbf7d0", "#86efac", 0, 0))
        parts.append(box(xv(70), y, xv(cur) - xv(70), 24, "#2563eb", "#2563eb", 0, 4))
        parts.append(text(xv(cur) + 8, y + 17, f"{cur:.2f}%", 11, anchor="start"))
        parts.append(text(xv(80), y + 42, "80", 10, fill="#166534"))
        parts.append(text(xv(85), y + 42, "85", 10, fill="#14532d"))
    parts.append(
        text(
            w / 2,
            h - 20,
            "Next: stable Track 1 ≥80% (multi-seed); lift clean811 method path; then multi-dataset.",
            11,
            fill="#374151",
        )
    )
    parts.append("</svg>")
    body = "\n".join(parts)
    write(FIG / "fig05_phase_a_track_progress.svg", body)
    write(RES / "phase_a_track_progress.svg", body)


if __name__ == "__main__":
    data_pipeline_svg()
    model_comparison_svg()
    results_bar_svg()
    complexity_svg()
    tracks_gap_svg()
    print("all figures OK")
