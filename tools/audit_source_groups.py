import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def default_data_dir():
    candidates = [
        REPO_ROOT / "data" / "raw" / "UrbanSound8K",
        REPO_ROOT.parents[2] / "data" / "UrbanSound8K",
    ]
    for candidate in candidates:
        if (candidate / "metadata" / "UrbanSound8K.csv").exists():
            return candidate
    return candidates[0]


def parse_folds(value):
    folds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    unique = []
    for fold in folds:
        if not 1 <= fold <= 10:
            raise ValueError(f"fold must be in [1, 10], got {fold}")
        if fold not in unique:
            unique.append(fold)
    return unique


def parse_classes(value):
    if value.strip().lower() == "all":
        return list(range(len(CLASS_NAMES)))
    by_name = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    class_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        class_id = int(part) if part.isdigit() else by_name.get(part)
        if class_id is None:
            raise ValueError(f"unknown class '{part}'")
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"class id must be in [0, 9], got {class_id}")
        if class_id not in class_ids:
            class_ids.append(class_id)
    return class_ids


def fmt_pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.2f}%"


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def load_metadata(data_dir):
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata" / "UrbanSound8K.csv"
    audio_base = data_dir / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    by_file = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["slice_file_name"]
            fold = int(row["fold"])
            by_file[file_name] = {
                "slice_file_name": file_name,
                "path": str(audio_base / f"fold{fold}" / file_name),
                "fsID": str(row["fsID"]),
                "classID": int(row["classID"]),
                "class": row["class"],
                "fold": fold,
                "salience": int(row["salience"]) if row.get("salience") else None,
                "start": float(row["start"]) if row.get("start") else None,
                "end": float(row["end"]) if row.get("end") else None,
            }
            if by_file[file_name]["start"] is not None and by_file[file_name]["end"] is not None:
                by_file[file_name]["duration"] = max(0.0, by_file[file_name]["end"] - by_file[file_name]["start"])
            else:
                by_file[file_name]["duration"] = None
    return by_file


def read_predictions(exp_name, fold, prediction_set):
    path = REPO_ROOT / "experiments" / exp_name / f"fold_{fold}" / "predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"missing predictions file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if prediction_set not in data:
        raise KeyError(f"{path} does not contain prediction set '{prediction_set}'")
    return data[prediction_set]


def build_group_rows(exp_name, folds, metadata, class_ids, prediction_set):
    groups = {}
    for fold in folds:
        predictions = read_predictions(exp_name, fold, prediction_set)
        for item in predictions:
            label = int(item["label"])
            if label not in class_ids:
                continue
            predicted = int(item["predicted"])
            file_name = Path(item["path"]).name
            meta = metadata.get(file_name, {})
            fsid = str(meta.get("fsID", file_name.split("-")[0]))
            class_id = int(meta.get("classID", label))
            key = (fold, label, fsid, class_id)
            row = groups.setdefault(
                key,
                {
                    "fold": fold,
                    "class_id": label,
                    "class_name": CLASS_NAMES[label],
                    "fsID": fsid,
                    "metadata_classID": class_id,
                    "support": 0,
                    "correct": 0,
                    "predicted_counts": collections.Counter(),
                    "files": [],
                    "wrong_files": [],
                    "official_folds": set(),
                    "salience_counts": collections.Counter(),
                    "durations": [],
                },
            )
            row["support"] += 1
            if predicted == label:
                row["correct"] += 1
            else:
                row["predicted_counts"][CLASS_NAMES[predicted]] += 1
                row["wrong_files"].append({"file": file_name, "predicted": CLASS_NAMES[predicted]})
            row["files"].append(file_name)
            if "fold" in meta:
                row["official_folds"].add(int(meta["fold"]))
            if meta.get("salience") is not None:
                row["salience_counts"][int(meta["salience"])] += 1
            if meta.get("duration") is not None:
                row["durations"].append(float(meta["duration"]))

    rows = []
    for row in groups.values():
        support = row["support"]
        accuracy = row["correct"] / support if support else None
        durations = row["durations"]
        rows.append(
            {
                "fold": row["fold"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "fsID": row["fsID"],
                "metadata_classID": row["metadata_classID"],
                "support": support,
                "correct": row["correct"],
                "accuracy_pct": 100.0 * accuracy if accuracy is not None else None,
                "top_confusions": [
                    {"predicted": name, "count": count}
                    for name, count in row["predicted_counts"].most_common(5)
                ],
                "files": row["files"],
                "wrong_files": row["wrong_files"],
                "official_folds": sorted(row["official_folds"]),
                "salience_counts": {str(k): v for k, v in sorted(row["salience_counts"].items())},
                "duration_mean": float(np.mean(durations)) if durations else None,
                "duration_min": float(np.min(durations)) if durations else None,
                "duration_max": float(np.max(durations)) if durations else None,
            }
        )

    rows.sort(key=lambda row: (row["accuracy_pct"], -row["support"], row["fold"], row["class_name"], row["fsID"]))
    return rows


def load_audio(path, sample_rate):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def audio_stats(file_names, metadata, sample_rate, max_files):
    stats = []
    for file_name in file_names[:max_files]:
        path = metadata[file_name]["path"]
        waveform = load_audio(path, sample_rate)
        x = waveform[0].float()
        rms = torch.sqrt(torch.mean(x * x).clamp_min(1e-12)).item()
        zcr = (x[:-1] * x[1:] < 0).float().mean().item() if x.numel() > 1 else 0.0
        spec = torch.stft(
            x,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=True,
        ).abs()
        freqs = torch.linspace(0, sample_rate / 2, spec.shape[0])
        mag = spec.mean(dim=1).clamp_min(1e-12)
        centroid = float((freqs * mag).sum().item() / mag.sum().item())
        stats.append(
            {
                "file": file_name,
                "rms_db": 20.0 * math.log10(max(rms, 1e-12)),
                "zero_crossing_rate": zcr,
                "spectral_centroid_hz": centroid,
            }
        )
    if not stats:
        return {}
    return {
        "rms_db_mean": float(np.mean([s["rms_db"] for s in stats])),
        "zero_crossing_rate_mean": float(np.mean([s["zero_crossing_rate"] for s in stats])),
        "spectral_centroid_hz_mean": float(np.mean([s["spectral_centroid_hz"] for s in stats])),
        "files_analyzed": len(stats),
        "file_stats": stats,
    }


def render_spectrogram(row, metadata, output_dir, sample_rate, max_clips):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = [item["file"] for item in row["wrong_files"][:max_clips]]
    if len(files) < max_clips:
        for file_name in row["files"]:
            if file_name not in files:
                files.append(file_name)
            if len(files) >= max_clips:
                break
    if not files:
        return None

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=64,
        f_min=40.0,
        f_max=float(sample_rate / 2),
        power=2.0,
    )

    cols = min(3, len(files))
    rows = int(math.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    wrong_pred = {item["file"]: item["predicted"] for item in row["wrong_files"]}
    for ax, file_name in zip(axes.ravel(), files):
        waveform = load_audio(metadata[file_name]["path"], sample_rate)
        features = mel(waveform).clamp_min(1e-10).log10()[0]
        ax.imshow(features.numpy(), origin="lower", aspect="auto", cmap="magma")
        pred = wrong_pred.get(file_name, "correct")
        duration = metadata[file_name].get("duration")
        duration_text = f"{duration:.2f}s" if duration is not None else "duration N/A"
        ax.set_title(f"{file_name}\npred={pred}, {duration_text}", fontsize=8)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    title = (
        f"fold {row['fold']} | {row['class_name']} | fsID {row['fsID']} | "
        f"{row['correct']}/{row['support']} correct"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = (
        f"fold{row['fold']}_{sanitize(row['class_name'])}_fsid{sanitize(row['fsID'])}"
        f"_acc{int(round(row['accuracy_pct']))}_n{row['support']}.png"
    )
    image_path = output_dir / image_name
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return str(image_path)


def write_report(report, output_md):
    lines = []
    lines.append(f"# Source Group Audit: {report['exp_name']}")
    lines.append("")
    lines.append(f"Prediction set: `{report['prediction_set']}`")
    lines.append("")
    lines.append(f"Folds: {', '.join(str(fold) for fold in report['folds'])}")
    lines.append("")
    lines.append("## Worst Source Groups")
    lines.append("")
    lines.append("| Rank | Fold | Class | fsID | Accuracy | Correct | Support | Main confusions | Official folds | Duration mean | Audio means |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---|---|---:|---|")
    for idx, row in enumerate(report["selected_groups"], start=1):
        confusions = ", ".join(f"{item['predicted']} {item['count']}" for item in row["top_confusions"]) or "none"
        audio = row.get("audio_stats") or {}
        audio_text = "N/A"
        if audio:
            audio_text = (
                f"rms {audio['rms_db_mean']:.1f} dB, "
                f"zcr {audio['zero_crossing_rate_mean']:.3f}, "
                f"centroid {audio['spectral_centroid_hz_mean']:.0f} Hz"
            )
        lines.append(
            f"| {idx} | {row['fold']} | {row['class_name']} | {row['fsID']} | "
            f"{fmt_pct(row['accuracy_pct'])} | {row['correct']} | {row['support']} | "
            f"{confusions} | {', '.join(str(v) for v in row['official_folds'])} | "
            f"{row['duration_mean']:.2f}s | {audio_text} |"
        )
    lines.append("")

    lines.append("## Spectrogram Contact Sheets")
    lines.append("")
    for row in report["selected_groups"]:
        image_path = row.get("spectrogram_path")
        lines.append(f"### Fold {row['fold']} - {row['class_name']} - fsID {row['fsID']}")
        lines.append("")
        confusions = ", ".join(f"{item['predicted']} {item['count']}" for item in row["top_confusions"]) or "none"
        lines.append(
            f"Accuracy: {fmt_pct(row['accuracy_pct'])} ({row['correct']}/{row['support']}); "
            f"confusions: {confusions}."
        )
        lines.append("")
        if image_path:
            rel_path = Path(image_path).relative_to(Path(output_md).parent)
            lines.append(f"![{row['class_name']} fsID {row['fsID']}]({rel_path.as_posix()})")
            lines.append("")
        wrong_examples = ", ".join(f"{item['file']}->{item['predicted']}" for item in row["wrong_files"][:8])
        lines.append(f"Wrong examples: {wrong_examples if wrong_examples else 'none'}")
        lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit worst UrbanSound8K source groups for a completed experiment.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--folds", default="1-3")
    parser.add_argument("--classes", default="all")
    parser.add_argument("--prediction_set", default="last_snapshot_predictions")
    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--max_groups", type=int, default=20)
    parser.add_argument("--max_spectrogram_clips", type=int, default=6)
    parser.add_argument("--max_audio_stat_files", type=int, default=12)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--data_dir", default=str(default_data_dir()))
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    folds = parse_folds(args.folds)
    class_ids = parse_classes(args.classes)
    metadata = load_metadata(args.data_dir)
    group_rows = build_group_rows(args.exp_name, folds, metadata, class_ids, args.prediction_set)

    selected = [
        row for row in group_rows
        if row["support"] >= args.min_support and row["accuracy_pct"] < 100.0
    ][: args.max_groups]

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "experiments" / args.exp_name / "source_audit"
    image_dir = output_dir / "spectrograms"
    for row in selected:
        row["audio_stats"] = audio_stats(row["files"], metadata, args.sample_rate, args.max_audio_stat_files)
        row["spectrogram_path"] = render_spectrogram(
            row,
            metadata,
            image_dir,
            args.sample_rate,
            args.max_spectrogram_clips,
        )

    report = {
        "exp_name": args.exp_name,
        "folds": folds,
        "prediction_set": args.prediction_set,
        "classes": [CLASS_NAMES[class_id] for class_id in class_ids],
        "min_support": args.min_support,
        "all_group_count": len(group_rows),
        "selected_groups": selected,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "source_group_audit.json"
    output_md = output_dir / "source_group_audit.md"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, output_md)
    print(f"Source audit JSON: {output_json}")
    print(f"Source audit report: {output_md}")


if __name__ == "__main__":
    main()
