import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.finetune_ast_teacher import Collator, UrbanSoundClipDataset, json_safe
from tools.source_safe_feature_probe import CLASS_NAMES, apply_smoke_subsets, fmt_pct, make_split, parse_folds, pct
from train import default_data_dir, format_teacher_checkpoint_path, source_label_overlap_summary
from src.data import parse_dataset


def record_cache_key(record):
    return f"fold{int(record['fold'])}/{record['slice_file_name']}"


def parse_split_names(value):
    valid = {"train", "val", "test"}
    names = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in valid:
            raise ValueError(f"Unsupported split '{item}'. Use comma-separated train,val,test.")
        if item not in names:
            names.append(item)
    if not names:
        raise ValueError("At least one split must be selected.")
    return names


@torch.no_grad()
def cache_split_logits(model, loader, records, device, amp):
    model.eval()
    logits_by_index = {}
    correct = 0
    total = 0
    started = time.time()
    for input_values, labels, indices in loader:
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(input_values=input_values).logits.float()
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()
        for local_index, row_logits in zip(indices.tolist(), logits.cpu()):
            logits_by_index[int(local_index)] = row_logits

    logits_by_key = {}
    labels_by_key = {}
    for index, record in enumerate(records):
        if index not in logits_by_index:
            raise RuntimeError(f"Missing teacher logits for dataset index {index}.")
        key = record_cache_key(record)
        if key in logits_by_key:
            raise RuntimeError(f"Duplicate cache key: {key}")
        logits_by_key[key] = logits_by_index[index]
        labels_by_key[key] = int(record["label"])

    return {
        "logits_by_key": logits_by_key,
        "labels_by_key": labels_by_key,
        "metrics": {
            "clips": len(records),
            "accuracy": correct / max(total, 1),
            "seconds": time.time() - started,
        },
    }


def cache_fold(args, fold):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Setup] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    csv_path = os.path.join(args.data_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")
    clip_records = parse_dataset(csv_path, audio_base, CLASS_NAMES)
    split = make_split(clip_records, fold, args.protocol, seed=args.seed)
    split = apply_smoke_subsets(split, args)
    if not split["uses_validation"] and "val" in args.split_names:
        raise ValueError(f"Protocol '{args.protocol}' does not have a validation split.")

    checkpoint_path = Path(format_teacher_checkpoint_path(args.teacher_checkpoint_template, fold))
    output_path = Path(format_teacher_checkpoint_path(args.output_template, fold))
    if output_path.exists() and not args.overwrite:
        print(f"[Skip] fold={fold} existing cache: {output_path}")
        return output_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"AST teacher checkpoint not found: {checkpoint_path}")

    cache_dir = Path(args.hf_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (REPO_ROOT / cache_dir).resolve()
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
    )
    model = AutoModelForAudioClassification.from_pretrained(str(checkpoint_path)).to(device)
    collator = Collator(extractor, args.sample_rate)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collator,
        "shuffle": False,
    }

    logits_by_key = {}
    labels_by_key = {}
    split_metrics = {}
    for split_name in args.split_names:
        records = split[split_name]
        dataset = UrbanSoundClipDataset(records, args.sample_rate)
        loader = DataLoader(dataset, **loader_kwargs)
        result = cache_split_logits(model, loader, records, device, args.amp)
        overlap = None
        if split_name == "test":
            overlap = source_label_overlap_summary(split["train"], split["test"])["count"]
        elif split_name == "val":
            overlap = source_label_overlap_summary(split["train"], split["val"])["count"]
        split_metrics[split_name] = result["metrics"] | {"train_source_overlap": overlap}
        logits_by_key.update(result["logits_by_key"])
        labels_by_key.update(result["labels_by_key"])
        print(
            f"[Cache] fold={fold} split={split_name} clips={len(records)} "
            f"teacher_acc={fmt_pct(pct(result['metrics']['accuracy']))} "
            f"seconds={result['metrics']['seconds']:.1f}"
        )

    dtype = torch.float16 if args.output_dtype == "float16" else torch.float32
    logits_by_key = {key: value.to(dtype=dtype) for key, value in logits_by_key.items()}
    metadata = {
        "cache_version": "ast_teacher_logits_v1",
        "fold": fold,
        "protocol": args.protocol,
        "seed": args.seed,
        "splits": args.split_names,
        "sample_rate": args.sample_rate,
        "class_names": CLASS_NAMES,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_model_name": args.model_name,
        "hf_cache_dir": str(cache_dir),
        "output_dtype": args.output_dtype,
        "counts": {name: len(split[name]) for name in ["train", "val", "test"]},
        "source_label_overlap": {
            "train_test": source_label_overlap_summary(split["train"], split["test"])["count"],
            "train_val": source_label_overlap_summary(split["train"], split["val"])["count"]
            if split["uses_validation"]
            else None,
        },
        "split_metrics": split_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "logits_by_key": logits_by_key,
            "labels_by_key": labels_by_key,
        },
        output_path,
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    report_path = output_path.with_suffix(".summary.md")
    report_path.write_text(build_report(output_path, metadata), encoding="utf-8")
    print(f"[Write] cache: {output_path}")
    print(f"[Write] metadata: {metadata_path}")
    print(f"[Write] report: {report_path}")
    return output_path


def build_report(output_path, metadata):
    lines = [
        f"# AST Teacher Logits Cache: Fold {metadata['fold']}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Cache | `{output_path}` |",
        f"| Protocol | `{metadata['protocol']}` |",
        f"| Splits | `{', '.join(metadata['splits'])}` |",
        f"| Teacher checkpoint | `{metadata['teacher_checkpoint']}` |",
        f"| Output dtype | `{metadata['output_dtype']}` |",
        "",
        "## Split Metrics",
        "",
        "| Split | Clips | Teacher accuracy | Seconds |",
        "|---|---:|---:|---:|",
    ]
    for split_name, row in metadata["split_metrics"].items():
        lines.append(
            f"| {split_name} | {row['clips']} | "
            f"{fmt_pct(pct(row['accuracy']))} | {row['seconds']:.1f} |"
        )
    lines.append("")
    lines.append(
        "Only attach this cache to student train frames. Validation and test metrics must remain label-only."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cache fine-tuned AST teacher logits for KV260 student distillation.")
    parser.add_argument("--data_dir", default=default_data_dir())
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", default=None, help="Optional comma/range list, e.g. 1,2,3 or 1-10.")
    parser.add_argument("--protocol", default="source_group_8_1_1")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--hf_cache_dir", default="experiments/smoke_ast_embedding_probe/hf_cache")
    parser.add_argument("--teacher_checkpoint_template", required=True)
    parser.add_argument("--output_template", default="experiments/ast_teacher_logits/fold_{fold}/teacher_logits.pt")
    parser.add_argument("--splits", default="train", help="Comma-separated subset of train,val,test. Default: train.")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--max_test_clips", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.split_names = parse_split_names(args.splits)

    folds = parse_folds(args.folds) if args.folds else [args.fold]
    for fold in folds:
        cache_fold(args, fold)


if __name__ == "__main__":
    main()
