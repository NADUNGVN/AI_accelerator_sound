import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_audio_to_ram, parse_dataset
from tools.source_safe_feature_probe import (
    CLASS_NAMES,
    apply_smoke_subsets,
    fmt_pct,
    make_split,
    pct,
    per_class_rows,
    weak_source_groups,
)
from train import default_data_dir, source_label_overlap_summary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def class_weights(records, device):
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    for record in records:
        counts[int(record["label"])] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return weights.to(device)


def build_weighted_sampler(records):
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    labels = []
    for record in records:
        label = int(record["label"])
        labels.append(label)
        counts[label] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    sample_weights = torch.tensor([float(weights[label]) for label in labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


class WaveformAugment:
    def __init__(self, enabled=False, gain_db=3.0, shift_prob=0.25, shift_max_fraction=0.08, noise_prob=0.10):
        self.enabled = bool(enabled)
        self.gain_db = float(gain_db)
        self.shift_prob = float(shift_prob)
        self.shift_max_fraction = float(shift_max_fraction)
        self.noise_prob = float(noise_prob)

    def __call__(self, waveform):
        if not self.enabled:
            return waveform
        x = waveform.copy()
        if self.gain_db > 0.0:
            gain = 10.0 ** (random.uniform(-self.gain_db, self.gain_db) / 20.0)
            x = x * gain
        if self.shift_prob > 0.0 and random.random() < self.shift_prob:
            max_shift = int(len(x) * self.shift_max_fraction)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                x = np.roll(x, shift)
                if shift > 0:
                    x[:shift] = 0.0
                elif shift < 0:
                    x[shift:] = 0.0
        if self.noise_prob > 0.0 and random.random() < self.noise_prob:
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            noise_rms = rms / (10.0 ** (random.uniform(24.0, 40.0) / 20.0))
            x = x + np.random.randn(*x.shape).astype(np.float32) * noise_rms
        return np.clip(x, -1.0, 1.0).astype(np.float32)


class UrbanSoundClipDataset(Dataset):
    def __init__(self, records, sample_rate, augment=None):
        self.records = list(records)
        self.sample_rate = int(sample_rate)
        self.augment = augment or WaveformAugment(enabled=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # load_audio_to_ram returns (path, waveform_numpy[C, T]) — not 3-tuple
        _, waveform = load_audio_to_ram(record["path"], self.sample_rate)
        waveform = np.asarray(waveform[0], dtype=np.float32)
        waveform = self.augment(waveform)
        return waveform, int(record["label"]), index


class Collator:
    def __init__(self, extractor, sample_rate):
        self.extractor = extractor
        self.sample_rate = int(sample_rate)

    def __call__(self, batch):
        waveforms, labels, indices = zip(*batch)
        inputs = self.extractor(
            list(waveforms),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        labels = torch.tensor(labels, dtype=torch.long)
        indices = torch.tensor(indices, dtype=torch.long)
        return inputs["input_values"], labels, indices


def set_base_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        if "classifier" not in name:
            parameter.requires_grad_(bool(trainable))


def build_optimizer(model, args):
    head_params = []
    encoder_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(parameter)
        else:
            encoder_params.append(parameter)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr, "base_lr": args.encoder_lr, "name": "encoder"})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr, "name": "head"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def get_lr_scale(epoch, warmup_epochs, total_epochs):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return max(1e-3, float(epoch + 1) / float(warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def train_epoch(model, loader, optimizer, scaler, criterion, device, args, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    optimizer.zero_grad(set_to_none=True)
    lr_scale = get_lr_scale(epoch, args.lr_warmup_epochs, total_epochs)
    for group in optimizer.param_groups:
        group["lr"] = max(args.min_lr, float(group["base_lr"]) * lr_scale)

    start_time = time.time()
    for step, (input_values, labels, _) in enumerate(loader):
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(input_values=input_values)
            logits = outputs.logits
            loss = criterion(logits.float(), labels) / args.accum_steps
        scaler.scale(loss).backward()

        is_boundary = (step + 1) % args.accum_steps == 0 or (step + 1) == len(loader)
        if is_boundary:
            scaler.unscale_(optimizer)
            if args.gradient_clip is not None and args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * args.accum_steps * labels.size(0)
        predicted = logits.argmax(dim=1)
        total += labels.numel()
        correct += int((predicted == labels).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "seconds": time.time() - start_time,
        "lr_scale": lr_scale,
        "head_lr": next((group["lr"] for group in optimizer.param_groups if group.get("name") == "head"), 0.0),
        "encoder_lr": next((group["lr"] for group in optimizer.param_groups if group.get("name") == "encoder"), 0.0),
    }


@torch.no_grad()
def evaluate(model, loader, records, device, args):
    model.eval()
    predictions_by_index = {}
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    for input_values, labels, indices in loader:
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(input_values=input_values)
            logits = outputs.logits.float()
            loss = criterion(logits, labels)
        predicted = logits.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.numel()
        correct += int((predicted == labels).sum().item())
        for local_index, pred in zip(indices.tolist(), predicted.cpu().tolist()):
            predictions_by_index[int(local_index)] = int(pred)

    y_true = [int(record["label"]) for record in records]
    y_pred = [predictions_by_index[idx] for idx in range(len(records))]
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "seconds": time.time() - start_time,
        "per_class": per_class_rows(y_true, y_pred),
        "weak_source_groups": weak_source_groups(records, y_pred, min_support=args.min_source_support, limit=args.weak_group_limit),
    }


def save_checkpoint(path, model, epoch, val_acc, test_acc, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    payload = {
        "epoch": epoch,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "args": vars(args),
    }
    (path / "teacher_state.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def load_checkpoint(path, device):
    model = AutoModelForAudioClassification.from_pretrained(path).to(device)
    model.eval()
    return model


def write_report(exp_dir, metrics):
    metrics_path = exp_dir / "metrics.json"
    report_path = exp_dir / "summary.md"
    metrics_path.write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")

    best = metrics["best"]
    final = metrics["final"]
    lines = [
        f"# AST Teacher Fine-Tune: {exp_dir.name}",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Completed epochs | {metrics['completed_epochs']} |",
        f"| Early stopped | {metrics['early_stopped']} |",
        f"| Best epoch | {best['epoch']} |",
        f"| Best val | {fmt_pct(pct(best['val']['accuracy']))} |",
        f"| Best test | {fmt_pct(pct(best['test']['accuracy']))} |",
        f"| Final val | {fmt_pct(pct(final['val']['accuracy']))} |",
        f"| Final test | {fmt_pct(pct(final['test']['accuracy']))} |",
        "",
        "## Best-Test Per Class",
        "",
        "| Class | Support | Correct | Accuracy | Main wrong |",
        "|---|---:|---:|---:|---|",
    ]
    for row in best["test"]["per_class"]:
        wrong = ", ".join(f"{item['predicted']}:{item['count']}" for item in row["top_confusions"])
        lines.append(
            f"| {row['class_name']} | {row['support']} | {row['correct']} | "
            f"{fmt_pct(pct(row['accuracy']))} | {wrong} |"
        )

    lines.extend(["", "## Weak Source Groups", ""])
    lines.extend(["| Class | fsID | Correct/support | Acc | Main wrong |", "|---|---:|---:|---:|---|"])
    for row in best["test"]["weak_source_groups"]:
        wrong = ", ".join(f"{item['predicted']}:{item['count']}" for item in row["top_wrong"])
        lines.append(
            f"| {row['class_name']} | {row['fsID']} | {row['correct']}/{row['support']} | "
            f"{fmt_pct(pct(row['accuracy']))} | {wrong} |"
        )

    lines.extend(["", "## History", ""])
    lines.extend(["| Epoch | Train | Val | Test | Train loss | Seconds |", "|---:|---:|---:|---:|---:|---:|"])
    for row in metrics["history"]:
        test_acc = row.get("test", {}).get("accuracy") if row.get("test") else None
        lines.append(
            f"| {row['epoch']} | {fmt_pct(pct(row['train']['accuracy']))} | "
            f"{fmt_pct(pct(row['val']['accuracy']))} | {fmt_pct(pct(test_acc))} | "
            f"{row['train']['loss']:.4f} | {row['train']['seconds']:.1f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AST teacher on source-safe UrbanSound8K splits.")
    parser.add_argument("--data_dir", default=default_data_dir())
    parser.add_argument("--exp_name", default="ast_teacher_finetune")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--protocol", default="source_group_8_1_1")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--hf_cache_dir", default="experiments/smoke_ast_embedding_probe/hf_cache")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--lr_warmup_epochs", type=int, default=2)
    parser.add_argument("--freeze_base_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--balanced_loss", action="store_true", default=True)
    parser.add_argument("--no_balanced_loss", action="store_false", dest="balanced_loss")
    parser.add_argument("--weighted_sampler", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_warmup", type=int, default=6)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
    parser.add_argument("--eval_test_each_epoch", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--max_test_clips", type=int, default=None)
    parser.add_argument("--min_source_support", type=int, default=5)
    parser.add_argument("--weak_group_limit", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    exp_dir = REPO_ROOT / "experiments" / args.exp_name / f"fold_{args.fold}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Setup] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    csv_path = os.path.join(args.data_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")
    clip_records = parse_dataset(csv_path, audio_base, CLASS_NAMES)
    split = make_split(clip_records, args.fold, args.protocol, seed=args.seed)
    split = apply_smoke_subsets(split, args)
    overlap = source_label_overlap_summary(split["train"], split["test"])
    val_overlap = source_label_overlap_summary(split["train"], split["val"]) if split["uses_validation"] else {"count": None}
    print(
        f"[Split] fold={args.fold} protocol={args.protocol} "
        f"clips train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} "
        f"train/test source-label overlap={overlap['count']} train/val source-label overlap={val_overlap['count']}"
    )
    if not split["uses_validation"] or not split["val"]:
        raise ValueError("AST fine-tune requires a validation split. Use source_group_8_1_1 or clean_8_1_1.")

    cache_dir = Path(args.hf_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (REPO_ROOT / cache_dir).resolve()
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
    )
    id2label = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_dataset = UrbanSoundClipDataset(
        split["train"],
        args.sample_rate,
        augment=WaveformAugment(enabled=args.augment),
    )
    val_dataset = UrbanSoundClipDataset(split["val"], args.sample_rate)
    test_dataset = UrbanSoundClipDataset(split["test"], args.sample_rate)
    collator = Collator(extractor, args.sample_rate)

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collator,
        "drop_last": False,
    }
    if args.weighted_sampler:
        train_loader_kwargs["sampler"] = build_weighted_sampler(split["train"])
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    eval_loader_kwargs = {
        "batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collator,
        "shuffle": False,
    }
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    weights = class_weights(split["train"], device) if args.balanced_loss else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    set_base_trainable(model, trainable=args.freeze_base_epochs <= 0)
    optimizer = build_optimizer(model, args)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        "[Model Setup] "
        f"model={args.model_name} | params={total_params:,} | trainable={trainable:,} | "
        f"device={device} | batch={args.batch_size} | accum={args.accum_steps}"
    )
    if weights is not None:
        print(f"[Loss Setup] balanced weights={[round(float(v), 4) for v in weights.cpu()]}")

    history = []
    best_val = -1.0
    best_payload = None
    epochs_without_improvement = 0
    early_stopped = False

    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        if epoch == args.freeze_base_epochs + 1:
            set_base_trainable(model, trainable=True)
            optimizer = build_optimizer(model, args)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Model Setup] unfreezing AST base at epoch {epoch}; trainable={trainable:,}")

        train_metrics = train_epoch(model, train_loader, optimizer, scaler, criterion, device, args, epoch_idx, args.epochs)
        val_metrics = evaluate(model, val_loader, split["val"], device, args)
        test_metrics = evaluate(model, test_loader, split["test"], device, args) if args.eval_test_each_epoch else None
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)
        test_text = fmt_pct(pct(test_metrics["accuracy"])) if test_metrics is not None else "deferred"
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train={fmt_pct(pct(train_metrics['accuracy']))} Loss={train_metrics['loss']:.4f} | "
            f"Val={fmt_pct(pct(val_metrics['accuracy']))} | "
            f"Test={test_text} | "
            f"Time={train_metrics['seconds']:.1f}s"
        )

        val_acc = float(val_metrics["accuracy"])
        if val_acc > best_val + args.early_stop_min_delta:
            best_val = val_acc
            epochs_without_improvement = 0
            best_payload = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }
            test_acc_for_state = test_metrics["accuracy"] if test_metrics is not None else None
            save_checkpoint(exp_dir / "checkpoints" / "best", model, epoch, val_acc, test_acc_for_state, args)
            print(f"--> Saved best checkpoint at epoch {epoch}.")
        else:
            epochs_without_improvement += 1

        if epoch >= args.early_stop_warmup and epochs_without_improvement >= args.early_stop_patience:
            early_stopped = True
            print(
                f"Early stopping at epoch {epoch}: no val improvement for "
                f"{epochs_without_improvement} epochs."
            )
            break

    final_payload = history[-1]
    final_test_metrics = evaluate(model, test_loader, split["test"], device, args)
    final_payload["test"] = final_test_metrics
    save_checkpoint(
        exp_dir / "checkpoints" / "final",
        model,
        final_payload["epoch"],
        final_payload["val"]["accuracy"],
        final_test_metrics["accuracy"],
        args,
    )
    best_model = load_checkpoint(exp_dir / "checkpoints" / "best", device)
    best_test_metrics = evaluate(best_model, test_loader, split["test"], device, args)
    best_payload["test"] = best_test_metrics
    metrics = {
        "exp_name": args.exp_name,
        "fold": args.fold,
        "protocol": args.protocol,
        "seed": args.seed,
        "model_name": args.model_name,
        "hf_cache_dir": str(cache_dir),
        "counts": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "source_label_overlap": {
            "train_test": overlap["count"],
            "train_val": val_overlap["count"],
        },
        "params": {
            "total": total_params,
            "trainable_last_epoch": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "completed_epochs": len(history),
        "planned_epochs": args.epochs,
        "early_stopped": early_stopped,
        "best": best_payload,
        "final": final_payload,
        "history": history,
    }
    metrics_path, report_path = write_report(exp_dir, metrics)
    print(f"\nMetrics written: {metrics_path}")
    print(f"Report written : {report_path}")


if __name__ == "__main__":
    main()
