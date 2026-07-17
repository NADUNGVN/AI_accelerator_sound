import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import load_audio_to_ram, parse_dataset
from src.models import TCAM1DCNN
from src.training import Trainer


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


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    model = TCAM1DCNN(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def preload_waveforms(records, sample_rate):
    cached = {}
    paths = sorted({r["path"] for r in records})
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path, waveform in executor.map(lambda p: load_audio_to_ram(p, sample_rate), paths):
            cached[path] = waveform
    return cached


def confusion_from_predictions(predictions):
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for item in predictions:
        label = int(item["label"])
        predicted = int(item["predicted"])
        matrix[label][predicted] += 1

    rows = []
    for idx, name in enumerate(CLASS_NAMES):
        support = sum(matrix[idx])
        correct = matrix[idx][idx]
        predicted_as = sum(matrix[row][idx] for row in range(len(CLASS_NAMES)))
        rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "support": support,
                "correct": correct,
                "accuracy": correct / support if support else None,
                "predicted_count": predicted_as,
                "top_confusions": top_confusions(matrix[idx], idx),
            }
        )
    return matrix, rows


def top_confusions(row, label_idx, limit=3):
    pairs = [
        (CLASS_NAMES[idx], count)
        for idx, count in enumerate(row)
        if idx != label_idx and count > 0
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"predicted": name, "count": count} for name, count in pairs[:limit]]


def print_class_table(title, rows):
    print(f"\n{title}")
    print("class_id  class_name            support  correct  acc%    pred_count  top_confusions")
    for row in rows:
        acc = "NA" if row["accuracy"] is None else f"{row['accuracy'] * 100:6.2f}"
        conf = ", ".join(f"{c['predicted']}:{c['count']}" for c in row["top_confusions"])
        print(
            f"{row['class_id']:>7}  {row['class_name']:<20} "
            f"{row['support']:>7}  {row['correct']:>7}  {acc}  "
            f"{row['predicted_count']:>10}  {conf}"
        )


def evaluate_split(name, trainer, models, records, cached_waveforms, frame_length):
    acc, predictions = trainer.evaluate_clips(
        models,
        records,
        cached_waveforms,
        frame_length=frame_length,
        return_predictions=True,
    )
    matrix, rows = confusion_from_predictions(predictions)
    print(f"\n{name} clip accuracy: {acc * 100:.2f}% ({len(predictions)} clips)")
    print_class_table(f"{name} per-class accuracy", rows)
    return {
        "accuracy": acc,
        "clip_count": len(predictions),
        "confusion_matrix": matrix,
        "per_class": rows,
    }


def grouped_clips(records):
    clips = collections.defaultdict(list)
    for record in records:
        clips[record["path"]].append(record)
    return clips


def predict_modes_for_clip(models, waveform_np, frame_length, device, zero_threshold):
    waveform = torch.from_numpy(waveform_np)
    frames = []
    valid = []
    for idx in range(15):
        offset = idx * (frame_length // 2)
        frame = waveform[:, offset : offset + frame_length]
        if frame.shape[-1] < frame_length:
            frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode="constant")
        frames.append(frame)
        valid.append(float(frame.abs().max().item()) > zero_threshold)

    if not any(valid):
        valid = [True for _ in frames]

    batch_tensor = torch.stack(frames).to(device)
    probs_sum = torch.zeros((len(frames), len(CLASS_NAMES)), device=device)
    with torch.no_grad():
        for model in models:
            logits = model(batch_tensor)
            probs_sum += F.softmax(logits, dim=-1)
    probs = probs_sum / len(models)

    valid_mask = torch.tensor(valid, dtype=torch.bool, device=device)
    all_frame_preds = probs.argmax(dim=1)
    valid_frame_preds = probs[valid_mask].argmax(dim=1)

    return {
        "sum_all": int(probs.sum(dim=0).argmax().item()),
        "majority_all": int(torch.bincount(all_frame_preds, minlength=len(CLASS_NAMES)).argmax().item()),
        "sum_nonzero": int(probs[valid_mask].sum(dim=0).argmax().item()),
        "majority_nonzero": int(torch.bincount(valid_frame_preds, minlength=len(CLASS_NAMES)).argmax().item()),
        "max_frame_conf": int(probs.reshape(-1).argmax().item() % len(CLASS_NAMES)),
        "valid_frame_count": int(valid_mask.sum().item()),
    }


def evaluate_split_modes(name, models, records, cached_waveforms, frame_length, device, zero_threshold):
    clips = grouped_clips(records)
    mode_predictions = {
        "sum_all": [],
        "majority_all": [],
        "sum_nonzero": [],
        "majority_nonzero": [],
        "max_frame_conf": [],
    }
    valid_frame_counts = []

    for path, frames in clips.items():
        label = frames[0]["label"]
        result = predict_modes_for_clip(
            models,
            cached_waveforms[path],
            frame_length,
            device,
            zero_threshold,
        )
        valid_frame_counts.append(result["valid_frame_count"])
        for mode in mode_predictions:
            mode_predictions[mode].append(
                {
                    "path": path,
                    "label": label,
                    "predicted": result[mode],
                }
            )

    print(f"\n{name} aggregation-mode comparison")
    print("mode              acc%    correct/total")
    report = {}
    total = len(clips)
    for mode, predictions in mode_predictions.items():
        correct = sum(1 for item in predictions if item["label"] == item["predicted"])
        matrix, rows = confusion_from_predictions(predictions)
        acc = correct / total if total else 0.0
        report[mode] = {
            "accuracy": acc,
            "clip_count": total,
            "correct": correct,
            "confusion_matrix": matrix,
            "per_class": rows,
        }
        print(f"{mode:<17} {acc * 100:6.2f}  {correct}/{total}")

    avg_valid = sum(valid_frame_counts) / len(valid_frame_counts) if valid_frame_counts else 0.0
    report["valid_frame_count"] = {
        "average": avg_valid,
        "min": min(valid_frame_counts) if valid_frame_counts else None,
        "max": max(valid_frame_counts) if valid_frame_counts else None,
        "zero_threshold": zero_threshold,
    }
    print(
        f"nonzero-frame count per clip: avg={avg_valid:.2f}, "
        f"min={report['valid_frame_count']['min']}, max={report['valid_frame_count']['max']}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze a completed TCAM1DCNN experiment without retraining.")
    parser.add_argument("--exp_dir", required=True, help="Experiment fold directory, e.g. experiments/paper9_msle_fp32/fold_1")
    parser.add_argument("--data_dir", default="data/raw/UrbanSound8K")
    parser.add_argument("--config", default="configs/rtx3090_config.json")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to cycle 4 inside exp_dir.")
    parser.add_argument("--eval_train", action="store_true", help="Also evaluate train clips with the selected checkpoint.")
    parser.add_argument("--ensemble_last2", action="store_true", help="Analyze ensemble of cycle 3 and cycle 4.")
    parser.add_argument("--eval_modes", action="store_true", help="Compare SUM, majority, and nonzero-frame aggregation modes.")
    parser.add_argument("--zero_threshold", type=float, default=1e-8, help="Frame max-abs threshold used by nonzero-frame aggregation.")
    args = parser.parse_args()

    cfg = read_json(args.config) or {}
    metrics = read_json(os.path.join(args.exp_dir, "metrics.json"))
    history = read_json(os.path.join(args.exp_dir, "history.json"))

    print("=== Experiment metadata ===")
    print(f"exp_dir: {args.exp_dir}")
    print(f"data_dir: {args.data_dir}")
    print(f"fold: {args.fold}")
    if metrics:
        keys = [
            "protocol",
            "loss_type",
            "amp",
            "gradient_clip",
            "adam_eps",
            "batch_size",
            "test_acc_last_snapshot",
            "test_acc_ensemble",
        ]
        for key in keys:
            print(f"{key}: {metrics.get(key)}")
    if history and history.get("train_acc"):
        print(f"final_train_frame_acc: {history['train_acc'][-1] * 100:.2f}%")

    csv_path = os.path.join(args.data_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")
    clip_records = parse_dataset(csv_path, audio_base, CLASS_NAMES)
    test_records = [r for r in clip_records if r["fold"] == args.fold]
    train_records = [r for r in clip_records if r["fold"] != args.fold]

    print("\n=== Split counts ===")
    print(f"train clips: {len(train_records)}")
    print(f"test clips : {len(test_records)}")

    selected_records = list(test_records)
    if args.eval_train:
        selected_records.extend(train_records)
    print(f"\nPreloading {len({r['path'] for r in selected_records})} unique clips...")
    cached_waveforms = preload_waveforms(selected_records, cfg.get("sample_rate", 16000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=None,
        optimizer=None,
        criterion=None,
        scaler=None,
        device=device,
        use_amp=bool(cfg.get("amp", False)),
        gradient_clip=cfg.get("gradient_clip", None),
    )

    if args.ensemble_last2:
        checkpoint_paths = [
            os.path.join(args.exp_dir, "checkpoints", f"tcam_fold_{args.fold}_cycle_3.pt"),
            os.path.join(args.exp_dir, "checkpoints", f"tcam_fold_{args.fold}_cycle_4.pt"),
        ]
        models = [load_model(path, device) for path in checkpoint_paths]
        model_name = "ensemble_last2"
    else:
        checkpoint_path = args.checkpoint or os.path.join(
            args.exp_dir,
            "checkpoints",
            f"tcam_fold_{args.fold}_cycle_4.pt",
        )
        models = [load_model(checkpoint_path, device)]
        model_name = os.path.basename(checkpoint_path)

    print(f"\n=== Evaluating {model_name} on {device} ===")
    report = {
        "exp_dir": args.exp_dir,
        "fold": args.fold,
        "model": model_name,
        "metrics": metrics,
        "final_train_frame_acc": history["train_acc"][-1] if history and history.get("train_acc") else None,
    }
    report["test"] = evaluate_split(
        "TEST",
        trainer,
        models,
        test_records,
        cached_waveforms,
        cfg.get("frame_length", 8000),
    )
    if args.eval_train:
        report["train"] = evaluate_split(
            "TRAIN",
            trainer,
            models,
            train_records,
            cached_waveforms,
            cfg.get("frame_length", 8000),
        )
    if args.eval_modes:
        report["test_modes"] = evaluate_split_modes(
            "TEST",
            models,
            test_records,
            cached_waveforms,
            cfg.get("frame_length", 8000),
            device,
            args.zero_threshold,
        )
        if args.eval_train:
            report["train_modes"] = evaluate_split_modes(
                "TRAIN",
                models,
                train_records,
                cached_waveforms,
                cfg.get("frame_length", 8000),
                device,
                args.zero_threshold,
            )

    output_path = os.path.join(args.exp_dir, f"analysis_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAnalysis written: {output_path}")


if __name__ == "__main__":
    main()
