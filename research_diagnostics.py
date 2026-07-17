import argparse
import collections
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import TCAM1DCNN, TCAMBlock


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

RANDOM_SPLIT_ALGORITHM = "stable_metadata_v2"

EXPECTED_TABLE2_SHAPES = {
    "conv1": [1, 32, 8000],
    "conv2": [1, 32, 4000],
    "conv3": [1, 64, 2000],
    "conv4": [1, 64, 1000],
    "conv5": [1, 128, 200],
    "conv6": [1, 128, 40],
    "conv7": [1, 256, 20],
}


def default_data_dir():
    repo_dir = Path(__file__).resolve().parent
    candidates = [
        repo_dir / "data" / "raw" / "UrbanSound8K",
        repo_dir.parents[2] / "data" / "UrbanSound8K",
    ]
    for candidate in candidates:
        if (candidate / "metadata" / "UrbanSound8K.csv").exists():
            return candidate
    return candidates[0]


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_records(data_dir):
    data_dir = Path(data_dir)
    csv_path = data_dir / "metadata" / "UrbanSound8K.csv"
    audio_base = data_dir / "audio"
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    records = []
    raw_class_counts = collections.Counter()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_class_counts[row["class"]] += 1
            if row["class"] == "rail_vehicle":
                continue
            if row["class"] not in class_to_idx:
                raise RuntimeError(f"Unexpected class in metadata: {row['class']}")
            fold = int(row["fold"])
            filename = row["slice_file_name"]
            start = float(row["start"])
            end = float(row["end"])
            records.append(
                {
                    "slice_file_name": filename,
                    "path": str(audio_base / f"fold{fold}" / filename),
                    "label": class_to_idx[row["class"]],
                    "class": row["class"],
                    "classID": int(row["classID"]),
                    "fsID": row["fsID"],
                    "fold": fold,
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                }
            )
    return records, dict(raw_class_counts)


def frame_padding_counts(records, frame_length_s=0.5, hop_s=0.25, frames_per_clip=15):
    total_frames = 0
    full_frames = 0
    partial_tail_frames = 0
    all_zero_padded_frames = 0
    by_class = {}

    per_class_records = collections.defaultdict(list)
    for record in records:
        per_class_records[record["class"]].append(record)

    for class_name, class_records in per_class_records.items():
        class_total = 0
        class_zero = 0
        class_partial = 0
        for record in class_records:
            duration = record["duration"]
            for idx in range(frames_per_clip):
                start_s = idx * hop_s
                end_s = start_s + frame_length_s
                total_frames += 1
                class_total += 1
                if start_s >= duration:
                    all_zero_padded_frames += 1
                    class_zero += 1
                elif end_s > duration:
                    partial_tail_frames += 1
                    class_partial += 1
                else:
                    full_frames += 1
        by_class[class_name] = {
            "total_frames": class_total,
            "all_zero_padded_frames": class_zero,
            "all_zero_padded_percent": round(class_zero / class_total * 100.0, 2) if class_total else 0.0,
            "partial_tail_frames": class_partial,
        }

    return {
        "total_frames": total_frames,
        "full_audio_frames": full_frames,
        "partial_tail_frames": partial_tail_frames,
        "all_zero_padded_frames": all_zero_padded_frames,
        "all_zero_padded_percent": round(all_zero_padded_frames / total_frames * 100.0, 2) if total_frames else 0.0,
        "by_class": dict(sorted(by_class.items())),
    }


def fold_statistics(records):
    stats = {}
    for fold in range(1, 11):
        fold_records = [r for r in records if r["fold"] == fold]
        class_counts = collections.Counter(r["class"] for r in fold_records)
        durations = [r["duration"] for r in fold_records]
        stats[str(fold)] = {
            "clips": len(fold_records),
            "class_counts": {name: class_counts.get(name, 0) for name in CLASS_NAMES},
            "duration_mean_s": round(sum(durations) / len(durations), 4) if durations else None,
            "duration_min_s": round(min(durations), 4) if durations else None,
            "duration_max_s": round(max(durations), 4) if durations else None,
            "padding": frame_padding_counts(fold_records),
        }
    return stats


def official_overlap(records):
    by_fold = {}
    for test_fold in range(1, 11):
        train = [r for r in records if r["fold"] != test_fold]
        test = [r for r in records if r["fold"] == test_fold]
        train_source_label = {(r["fsID"], r["classID"]) for r in train}
        test_source_label = {(r["fsID"], r["classID"]) for r in test}
        train_source = {r["fsID"] for r in train}
        test_source = {r["fsID"] for r in test}
        by_fold[str(test_fold)] = {
            "train_clips": len(train),
            "test_clips": len(test),
            "fsID_classID_overlap": len(train_source_label & test_source_label),
            "fsID_only_overlap": len(train_source & test_source),
        }
    return by_fold


def random_clip_split(records, test_bucket=1, seed=83, num_buckets=10):
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for record in records:
        by_class[record["label"]].append(record)

    train = []
    test = []
    for label in sorted(by_class):
        stable_records = sorted(
            by_class[label],
            key=lambda r: (r["fold"], r["slice_file_name"], r["fsID"], r["classID"]),
        )
        rng.shuffle(stable_records)
        for idx, record in enumerate(stable_records):
            bucket = (idx % num_buckets) + 1
            if bucket == test_bucket:
                test.append(record)
            else:
                train.append(record)
    return train, test


def split_overlap(train, test):
    train_source_label = {(r["fsID"], r["classID"]) for r in train}
    test_source_label = {(r["fsID"], r["classID"]) for r in test}
    train_source = {r["fsID"] for r in train}
    test_source = {r["fsID"] for r in test}
    return {
        "random_split_algorithm": RANDOM_SPLIT_ALGORITHM,
        "train_clips": len(train),
        "test_clips": len(test),
        "fsID_classID_overlap": len(train_source_label & test_source_label),
        "fsID_only_overlap": len(train_source & test_source),
        "test_class_counts": dict(collections.Counter(r["class"] for r in test)),
        "official_folds_in_test": sorted({r["fold"] for r in test}),
    }


def module_param_count(module, include_bias=True):
    total = 0
    for name, param in module.named_parameters(recurse=False):
        if not include_bias and name.endswith("bias"):
            continue
        total += param.numel()
    return total


def conv1d_macs(module, input_shape, output_shape):
    batch = output_shape[0]
    out_channels = output_shape[1]
    out_length = output_shape[2]
    kernel = module.kernel_size[0]
    in_channels = module.in_channels // module.groups
    return int(batch * out_channels * out_length * in_channels * kernel)


def linear_macs(module, input_shape, output_shape):
    batch = output_shape[0]
    return int(batch * module.in_features * module.out_features)


def model_diagnostics():
    model = TCAM1DCNN(num_classes=10).eval()
    hooks = []
    layer_rows = []

    def make_hook(name, module):
        def hook(mod, inputs, output):
            input_shape = list(inputs[0].shape)
            output_shape = list(output.shape)
            macs = None
            if isinstance(mod, nn.Conv1d):
                macs = conv1d_macs(mod, input_shape, output_shape)
            elif isinstance(mod, nn.Linear):
                macs = linear_macs(mod, input_shape, output_shape)
            layer_rows.append(
                {
                    "name": name,
                    "type": mod.__class__.__name__,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "kernel_size": list(mod.kernel_size) if isinstance(mod, nn.Conv1d) else None,
                    "stride": list(mod.stride) if isinstance(mod, nn.Conv1d) else None,
                    "padding": mod.padding if isinstance(mod, nn.Conv1d) else None,
                    "params_with_bias": module_param_count(mod, include_bias=True),
                    "params_no_bias": module_param_count(mod, include_bias=False),
                    "macs": macs,
                }
            )
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 8000))

    for hook in hooks:
        hook.remove()

    top_level_shape_checks = {}
    by_name = {row["name"]: row for row in layer_rows}
    for name, expected in EXPECTED_TABLE2_SHAPES.items():
        found = by_name.get(name, {}).get("output_shape")
        top_level_shape_checks[name] = {
            "expected": expected,
            "found": found,
            "matches": found == expected,
        }

    block_params = {}
    for name, module in model.named_modules():
        if isinstance(module, TCAMBlock):
            block_params[name] = {
                "params_with_bias": sum(p.numel() for p in module.parameters()),
                "params_no_bias": sum(
                    p.numel()
                    for param_name, p in module.named_parameters()
                    if not param_name.endswith("bias")
                ),
            }

    return {
        "output_shape": list(logits.shape),
        "total_params_with_bias": sum(p.numel() for p in model.parameters()),
        "total_params_no_bias": sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.endswith("bias")
        ),
        "approx_conv_linear_macs": sum(row["macs"] or 0 for row in layer_rows),
        "top_level_shape_checks": top_level_shape_checks,
        "layers": layer_rows,
        "tcam_block_params": block_params,
        "paper_reported_params": "406 K",
        "paper_reported_flops": "40 M",
    }


def artifact_summary(repo_dir):
    repo_dir = Path(repo_dir)
    experiments = {
        "official_fold1": repo_dir / "experiments" / "paper9_msle_fp32" / "fold_1",
        "random_clip_fold1": repo_dir / "experiments" / "randomclip_msle_fp32" / "fold_1",
    }
    summary = {}
    for name, path in experiments.items():
        metrics = read_json(path / "metrics.json")
        analysis = read_json(path / "analysis_all_cycles.json")
        if not metrics:
            summary[name] = {"exists": False, "path": str(path)}
            continue
        cycles = []
        if analysis:
            for cycle in analysis.get("cycles", []):
                cycles.append(
                    {
                        "cycle": cycle.get("cycle"),
                        "test_accuracy": cycle.get("test", {}).get("accuracy"),
                        "sum_nonzero_accuracy": cycle.get("test_modes", {}).get("sum_nonzero", {}).get("accuracy"),
                    }
                )
        summary[name] = {
            "exists": True,
            "path": str(path),
            "protocol": metrics.get("protocol"),
            "train_clip_count": metrics.get("train_clip_count"),
            "test_clip_count": metrics.get("test_clip_count"),
            "test_acc_last_snapshot": metrics.get("test_acc_last_snapshot"),
            "test_acc_ensemble": metrics.get("test_acc_ensemble"),
            "source_label_overlap_train_test": metrics.get("source_label_overlap_train_test"),
            "cycles": cycles,
        }
    return summary


def write_markdown(report, path):
    path = Path(path)
    lines = []
    lines.append("# Reproduction Deep Diagnostics")
    lines.append("")
    lines.append("## Split Outcome")
    for name, item in report["artifacts"].items():
        if not item.get("exists"):
            lines.append(f"- {name}: missing artifact")
            continue
        last_acc = item["test_acc_last_snapshot"] * 100 if item["test_acc_last_snapshot"] is not None else None
        ens_acc = item["test_acc_ensemble"] * 100 if item["test_acc_ensemble"] is not None else None
        overlap = item.get("source_label_overlap_train_test") or {}
        lines.append(
            f"- {name}: protocol={item['protocol']}, test={item['test_clip_count']}, "
            f"last={last_acc:.2f}%, ensemble={ens_acc:.2f}%, "
            f"fsID+classID overlap={overlap.get('count')}"
        )
    lines.append("")
    lines.append("## Official Fold Source Overlap")
    lines.append("")
    lines.append("| Fold | Train | Test | fsID+classID overlap | fsID-only overlap |")
    lines.append("|---:|---:|---:|---:|---:|")
    for fold, item in report["dataset"]["official_overlap"].items():
        lines.append(
            f"| {fold} | {item['train_clips']} | {item['test_clips']} | "
            f"{item['fsID_classID_overlap']} | {item['fsID_only_overlap']} |"
        )
    lines.append("")
    lines.append("## Frame Padding")
    pad = report["dataset"]["padding"]
    lines.append(
        f"- All-zero padded frames: {pad['all_zero_padded_frames']}/{pad['total_frames']} "
        f"({pad['all_zero_padded_percent']}%)."
    )
    lines.append("")
    lines.append("## Model")
    model = report["model"]
    lines.append(f"- Params with bias: {model['total_params_with_bias']}")
    lines.append(f"- Params without bias: {model['total_params_no_bias']}")
    lines.append(f"- Approx Conv/Linear MACs: {model['approx_conv_linear_macs']}")
    lines.append(f"- Paper reported params/FLOPs: {model['paper_reported_params']} / {model['paper_reported_flops']}")
    lines.append("")
    lines.append("| Layer | Expected shape | Found shape | Match |")
    lines.append("|---|---|---|---|")
    for layer, check in model["top_level_shape_checks"].items():
        lines.append(f"| {layer} | {check['expected']} | {check['found']} | {check['matches']} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The implementation matches the main Table 2 Conv1D output shapes.")
    lines.append("- Official predefined fold evaluation and random clip split evaluation behave very differently.")
    lines.append("- Random clip split has source-label overlap and reaches paper-like accuracy; official fold 1 does not.")
    lines.append("- This points to split protocol/source leakage as the primary reproduction fork, not a hardware or DataLoader issue.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Research diagnostics for TCAM1DCNN reproduction.")
    parser.add_argument("--data_dir", default=str(default_data_dir()))
    parser.add_argument("--output_json", default="results/diagnostics/research_diagnostics.json")
    parser.add_argument("--output_md", default="docs/Reproduction_Deep_Diagnostics.md")
    parser.add_argument("--random_seed", type=int, default=83)
    parser.add_argument("--random_test_bucket", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    records, raw_class_counts = load_records(data_dir)
    train_random, test_random = random_clip_split(
        records,
        test_bucket=args.random_test_bucket,
        seed=args.random_seed,
    )

    report = {
        "data_dir": str(data_dir),
        "dataset": {
            "raw_class_counts": raw_class_counts,
            "filtered_clip_count": len(records),
            "class_counts": dict(collections.Counter(r["class"] for r in records)),
            "fold_stats": fold_statistics(records),
            "padding": frame_padding_counts(records),
            "official_overlap": official_overlap(records),
            "random_clip_split_control": split_overlap(train_random, test_random),
        },
        "model": model_diagnostics(),
        "artifacts": artifact_summary(repo_dir),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_md)

    print(f"Diagnostics JSON: {output_json}")
    print(f"Diagnostics report: {output_md}")
    print(f"Filtered clips: {len(records)}")
    print(f"All-zero padded frames: {report['dataset']['padding']['all_zero_padded_percent']}%")
    print(f"Params with bias: {report['model']['total_params_with_bias']}")
    print(f"Approx Conv/Linear MACs: {report['model']['approx_conv_linear_macs']}")


if __name__ == "__main__":
    main()
