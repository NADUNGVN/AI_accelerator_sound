import argparse
import collections
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import LogMelFeatureExtractor, load_audio_to_ram, parse_dataset
from src.models import TCAM1DCNN, EfficientAudioCNN1D, KV260AudioNetDS1D, KV260LogMelNetDS1D
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


RANDOM_SPLIT_ALGORITHM = "stable_metadata_v2"
SOURCE_GROUP_SPLIT_ALGORITHM = "fsid_classid_balanced_v1"


def default_data_dir():
    candidates = [
        os.path.join(REPO_ROOT, "data", "raw", "UrbanSound8K"),
        os.path.abspath(os.path.join(REPO_ROOT, "..", "..", "..", "data", "UrbanSound8K")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "metadata", "UrbanSound8K.csv")):
            return candidate
    return candidates[0]


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg, metrics, num_classes):
    model_name = (metrics or {}).get("model_name", cfg.get("model_name", "tcam1dcnn")).lower()
    if model_name == "tcam1dcnn":
        return TCAM1DCNN(num_classes=num_classes)
    if model_name == "efficient_audio_cnn1d":
        return EfficientAudioCNN1D(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.25)),
        )
    if model_name == "kv260_audio_net_ds1d":
        return KV260AudioNetDS1D(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.15)),
            pool_type=cfg.get("pool_type", "avg"),
            pool_bins=cfg.get("pool_bins", None),
            stem_type=cfg.get("stem_type", "single"),
        )
    if model_name == "kv260_logmel_net_ds1d":
        return KV260LogMelNetDS1D(
            num_classes=num_classes,
            input_channels=int(cfg.get("n_mels", 64)),
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.20)),
            pool_type=cfg.get("pool_type", "avgmax"),
        )
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def build_input_transform(cfg, device):
    input_features = cfg.get("input_features", "waveform").lower()
    if input_features == "waveform":
        return None
    if input_features == "logmel":
        return LogMelFeatureExtractor(
            sample_rate=cfg.get("sample_rate", 16000),
            n_fft=cfg.get("n_fft", 1024),
            hop_length=cfg.get("mel_hop_length", 256),
            win_length=cfg.get("win_length", cfg.get("n_fft", 1024)),
            n_mels=cfg.get("n_mels", 64),
            f_min=cfg.get("f_min", 40.0),
            f_max=cfg.get("f_max", None),
            eps=cfg.get("logmel_eps", 1e-6),
            normalize=cfg.get("logmel_normalize", True),
        ).to(device).eval()
    raise ValueError(f"Unsupported input_features '{input_features}'. Use 'waveform' or 'logmel'.")


def frame_settings(cfg, metrics):
    frame_length = int((metrics or {}).get("frame_length", cfg.get("frame_length", 8000)))
    frame_hop = int((metrics or {}).get("frame_hop", cfg.get("frame_hop", frame_length // 2)))
    frames_per_clip = int((metrics or {}).get("frames_per_clip", cfg.get("frames_per_clip", 15)))
    return frame_length, frame_hop, frames_per_clip


def eval_drop_silent_tail_frames(cfg, metrics):
    default_value = cfg.get("eval_drop_silent_tail_frames", cfg.get("drop_silent_tail_frames", False))
    return bool((metrics or {}).get("eval_drop_silent_tail_frames", default_value))


def load_model(checkpoint_path, device, cfg, metrics):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    model = build_model(cfg, metrics, num_classes=len(CLASS_NAMES)).to(device)
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


def random_split_sort_key(record, algorithm):
    if algorithm == "path_v1":
        return record["path"]
    return (
        record["label"],
        record["fold"],
        record.get("slice_file_name", os.path.basename(record["path"])),
        str(record.get("fsID", "")),
        int(record.get("classID", -1)),
    )


def make_stratified_clip_subset(records, max_clips, seed, algorithm=RANDOM_SPLIT_ALGORITHM):
    if max_clips is None or max_clips >= len(records):
        return records
    if max_clips <= 0:
        raise ValueError(f"max_clips must be positive when provided, got {max_clips}")

    rng = random.Random(seed)
    by_label = collections.defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    for label in by_label:
        by_label[label] = sorted(by_label[label], key=lambda r: random_split_sort_key(r, algorithm))
        rng.shuffle(by_label[label])

    selected = []
    label_order = sorted(by_label)
    while len(selected) < max_clips:
        progressed = False
        for label in label_order:
            if by_label[label]:
                selected.append(by_label[label].pop())
                progressed = True
                if len(selected) == max_clips:
                    break
        if not progressed:
            break
    return selected


def make_stratified_random_clip_split(clip_records, test_bucket, seed, algorithm=RANDOM_SPLIT_ALGORITHM, num_buckets=10):
    if not 1 <= test_bucket <= num_buckets:
        raise ValueError(f"test_bucket must be in [1, {num_buckets}], got {test_bucket}")

    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for record in clip_records:
        by_class[record["label"]].append(record)

    train_records = []
    test_records = []
    for label in sorted(by_class):
        records = sorted(by_class[label], key=lambda r: random_split_sort_key(r, algorithm))
        rng.shuffle(records)
        for idx, record in enumerate(records):
            bucket = (idx % num_buckets) + 1
            if bucket == test_bucket:
                test_records.append(record)
            else:
                train_records.append(record)

    return train_records, test_records


def make_stratified_source_group_split(clip_records, test_bucket, seed, num_buckets=10):
    if not 1 <= test_bucket <= num_buckets:
        raise ValueError(f"test_bucket must be in [1, {num_buckets}], got {test_bucket}")
    if not clip_records or "fsID" not in clip_records[0] or "classID" not in clip_records[0]:
        raise ValueError("source_group_9_1 requires fsID and classID metadata fields.")

    buckets_by_class = make_stratified_source_group_buckets(clip_records, seed, num_buckets=num_buckets)
    train_records = []
    test_records = []
    for label_buckets in buckets_by_class.values():
        for idx, records in enumerate(label_buckets, start=1):
            if idx == test_bucket:
                test_records.extend(records)
            else:
                train_records.extend(records)

    return train_records, test_records


def make_stratified_source_group_buckets(clip_records, seed, num_buckets=10):
    if not clip_records or "fsID" not in clip_records[0] or "classID" not in clip_records[0]:
        raise ValueError("source-group split requires fsID and classID metadata fields.")

    rng = random.Random(seed)
    groups_by_class = collections.defaultdict(dict)
    for record in clip_records:
        key = (str(record["fsID"]), int(record["classID"]))
        groups_by_class[record["label"]].setdefault(key, []).append(record)

    buckets_by_class = {}
    for label in sorted(groups_by_class):
        groups = sorted(
            groups_by_class[label].items(),
            key=lambda item: (
                min(r["fold"] for r in item[1]),
                item[0][0],
                item[0][1],
                min(r.get("slice_file_name", os.path.basename(r["path"])) for r in item[1]),
            ),
        )
        rng.shuffle(groups)
        groups.sort(key=lambda item: len(item[1]), reverse=True)

        buckets = [[] for _ in range(num_buckets)]
        bucket_counts = [0 for _ in range(num_buckets)]
        for _, records in groups:
            min_count = min(bucket_counts)
            candidates = [idx for idx, count in enumerate(bucket_counts) if count == min_count]
            bucket_idx = rng.choice(candidates)
            buckets[bucket_idx].extend(records)
            bucket_counts[bucket_idx] += len(records)

        buckets_by_class[label] = buckets

    return buckets_by_class


def make_stratified_source_group_train_val_test_split(clip_records, test_bucket, seed, num_buckets=10):
    val_bucket = (test_bucket % num_buckets) + 1
    buckets_by_class = make_stratified_source_group_buckets(clip_records, seed, num_buckets=num_buckets)

    train_records = []
    val_records = []
    test_records = []
    for label_buckets in buckets_by_class.values():
        for idx, records in enumerate(label_buckets, start=1):
            if idx == test_bucket:
                test_records.extend(records)
            elif idx == val_bucket:
                val_records.extend(records)
            else:
                train_records.extend(records)

    return train_records, val_records, test_records, val_bucket


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


def evaluate_split(
    name,
    trainer,
    models,
    records,
    cached_waveforms,
    frame_length,
    frame_hop,
    frames_per_clip,
    drop_silent_tail_frames=False,
    sample_rate=16000,
):
    acc, predictions = trainer.evaluate_clips(
        models,
        records,
        cached_waveforms,
        frame_length=frame_length,
        frame_hop=frame_hop,
        frames_per_clip=frames_per_clip,
        drop_silent_tail_frames=drop_silent_tail_frames,
        sample_rate=sample_rate,
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


def predict_modes_for_clip(
    models,
    waveform_np,
    frame_length,
    frame_hop,
    frames_per_clip,
    device,
    zero_threshold,
    duration_seconds=None,
    drop_silent_tail_frames=False,
    sample_rate=16000,
    input_transform=None,
):
    waveform = torch.from_numpy(waveform_np)
    frames = []
    valid = []
    duration_samples = None
    if drop_silent_tail_frames and duration_seconds is not None:
        duration_samples = int(float(duration_seconds) * sample_rate)
    for idx in range(frames_per_clip):
        offset = idx * frame_hop
        if duration_samples is not None and offset >= duration_samples:
            continue
        frame = waveform[:, offset : offset + frame_length]
        if frame.shape[-1] < frame_length:
            frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode="constant")
        frames.append(frame)
        valid.append(float(frame.abs().max().item()) > zero_threshold)

    if not frames:
        frame = waveform[:, :frame_length]
        if frame.shape[-1] < frame_length:
            frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode="constant")
        frames.append(frame)
        valid.append(float(frame.abs().max().item()) > zero_threshold)

    if not any(valid):
        valid = [True for _ in frames]

    batch_tensor = torch.stack(frames).to(device)
    if input_transform is not None:
        batch_tensor = input_transform(batch_tensor.float())
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


def evaluate_split_modes(
    name,
    models,
    records,
    cached_waveforms,
    frame_length,
    frame_hop,
    frames_per_clip,
    device,
    zero_threshold,
    drop_silent_tail_frames=False,
    sample_rate=16000,
    input_transform=None,
):
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
            frame_hop,
            frames_per_clip,
            device,
            zero_threshold,
            duration_seconds=frames[0].get("duration"),
            drop_silent_tail_frames=drop_silent_tail_frames,
            sample_rate=sample_rate,
            input_transform=input_transform,
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
    parser.add_argument("--data_dir", default=default_data_dir())
    parser.add_argument("--config", default="configs/rtx3090_config.json")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to cycle 4 inside exp_dir.")
    parser.add_argument("--eval_train", action="store_true", help="Also evaluate train clips with the selected checkpoint.")
    parser.add_argument("--ensemble_last2", action="store_true", help="Analyze ensemble of cycle 3 and cycle 4.")
    parser.add_argument("--eval_all_cycles", action="store_true", help="Evaluate every cycle checkpoint found in exp_dir.")
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
    protocol = (metrics or {}).get("protocol", cfg.get("protocol", "paper_9_1"))
    val_records = []
    if protocol == "random_clip_9_1":
        split_seed = (metrics or {}).get("seed", cfg.get("seed", 83))
        split_fold = (metrics or {}).get("test_fold", args.fold)
        if metrics and "random_split_algorithm" not in metrics:
            split_algorithm = "path_v1"
        else:
            split_algorithm = (metrics or {}).get("random_split_algorithm", cfg.get("random_split_algorithm", RANDOM_SPLIT_ALGORITHM))
        train_records, test_records = make_stratified_random_clip_split(
            clip_records,
            test_bucket=split_fold,
            seed=split_seed,
            algorithm=split_algorithm,
        )
        print(
            f"Reconstructed random_clip_9_1 split with test bucket={split_fold}, "
            f"seed={split_seed}, split_algorithm={split_algorithm}."
        )
    elif protocol == "source_group_9_1":
        split_seed = (metrics or {}).get("seed", cfg.get("seed", 83))
        split_fold = (metrics or {}).get("test_fold", args.fold)
        train_records, test_records = make_stratified_source_group_split(
            clip_records,
            test_bucket=split_fold,
            seed=split_seed,
        )
        print(
            f"Reconstructed source_group_9_1 split with test bucket={split_fold}, "
            f"seed={split_seed}, split_algorithm={SOURCE_GROUP_SPLIT_ALGORITHM}."
        )
    elif protocol == "source_group_8_1_1":
        split_seed = (metrics or {}).get("seed", cfg.get("seed", 83))
        split_fold = (metrics or {}).get("test_fold", args.fold)
        train_records, val_records, test_records, val_bucket = make_stratified_source_group_train_val_test_split(
            clip_records,
            test_bucket=split_fold,
            seed=split_seed,
        )
        print(
            f"Reconstructed source_group_8_1_1 split with test bucket={split_fold}, "
            f"val bucket={val_bucket}, seed={split_seed}, split_algorithm={SOURCE_GROUP_SPLIT_ALGORITHM}."
        )
    elif protocol == "clean_8_1_1":
        val_fold = (metrics or {}).get("val_fold", (args.fold % 10) + 1)
        test_records = [r for r in clip_records if r["fold"] == args.fold]
        train_records = [r for r in clip_records if r["fold"] != args.fold and r["fold"] != val_fold]
        val_records = [r for r in clip_records if r["fold"] == val_fold]
        print(f"Reconstructed clean_8_1_1 split with test fold={args.fold}, val fold={val_fold}.")
    else:
        test_records = [r for r in clip_records if r["fold"] == args.fold]
        train_records = [r for r in clip_records if r["fold"] != args.fold]
        print(f"Reconstructed paper_9_1 official split with test fold={args.fold}.")

    if metrics and (
        metrics.get("max_train_clips") is not None
        or metrics.get("max_val_clips") is not None
        or metrics.get("max_test_clips") is not None
    ):
        seed = metrics.get("seed", cfg.get("seed", 83))
        original_counts = (len(train_records), len(test_records))
        train_records = make_stratified_clip_subset(
            train_records,
            metrics.get("max_train_clips"),
            seed + 101,
            algorithm=(metrics.get("random_split_algorithm") or RANDOM_SPLIT_ALGORITHM),
        )
        test_records = make_stratified_clip_subset(
            test_records,
            metrics.get("max_test_clips"),
            seed + 303,
            algorithm=(metrics.get("random_split_algorithm") or RANDOM_SPLIT_ALGORITHM),
        )
        val_records = make_stratified_clip_subset(
            val_records,
            metrics.get("max_val_clips"),
            seed + 202,
            algorithm=(metrics.get("random_split_algorithm") or RANDOM_SPLIT_ALGORITHM),
        )
        print(
            "Reapplied smoke subset from metrics | "
            f"Train {original_counts[0]}->{len(train_records)}, "
            f"Test {original_counts[1]}->{len(test_records)}."
        )

    print("\n=== Split counts ===")
    print(f"train clips: {len(train_records)}")
    print(f"val clips  : {len(val_records)}")
    print(f"test clips : {len(test_records)}")

    selected_records = list(test_records)
    selected_records.extend(val_records)
    if args.eval_train:
        selected_records.extend(train_records)
    print(f"\nPreloading {len({r['path'] for r in selected_records})} unique clips...")
    cached_waveforms = preload_waveforms(selected_records, cfg.get("sample_rate", 16000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_transform = build_input_transform(cfg, device)
    trainer = Trainer(
        model=None,
        optimizer=None,
        criterion=None,
        scaler=None,
        device=device,
        use_amp=bool(cfg.get("amp", False)),
        gradient_clip=cfg.get("gradient_clip", None),
        input_transform=input_transform,
    )
    frame_length, frame_hop, frames_per_clip = frame_settings(cfg, metrics)
    drop_eval_tail = eval_drop_silent_tail_frames(cfg, metrics)
    sample_rate = int(cfg.get("sample_rate", 16000))
    print(
        f"Frame settings: frame_length={frame_length}, "
        f"frame_hop={frame_hop}, frames_per_clip={frames_per_clip}, "
        f"eval_drop_silent_tail_frames={drop_eval_tail}"
    )

    if args.eval_all_cycles:
        cycle_paths = [
            (
                cycle_id,
                os.path.join(args.exp_dir, "checkpoints", f"tcam_fold_{args.fold}_cycle_{cycle_id}.pt"),
            )
            for cycle_id in range(1, int(cfg.get("cycles", 4)) + 1)
        ]
        final_path = os.path.join(args.exp_dir, "checkpoints", f"tcam_fold_{args.fold}_cycle_final.pt")
        if os.path.exists(final_path):
            cycle_paths.append(("final", final_path))
        print(f"\n=== Evaluating all cycle checkpoints on {device} ===")
        report = {
            "exp_dir": args.exp_dir,
            "fold": args.fold,
            "model": "all_cycles",
            "metrics": metrics,
            "final_train_frame_acc": history["train_acc"][-1] if history and history.get("train_acc") else None,
            "cycles": [],
        }
        for cycle_id, checkpoint_path in cycle_paths:
            if not os.path.exists(checkpoint_path):
                print(f"\nCycle {cycle_id}: checkpoint missing, skipped: {checkpoint_path}")
                continue
            models = [load_model(checkpoint_path, device, cfg, metrics)]
            print(f"\n--- Cycle {cycle_id}: {os.path.basename(checkpoint_path)} ---")
            cycle_report = {
                "cycle": cycle_id,
                "checkpoint": checkpoint_path,
                "test": evaluate_split(
                    "TEST",
                    trainer,
                    models,
                    test_records,
                    cached_waveforms,
                    frame_length,
                    frame_hop,
                    frames_per_clip,
                    drop_eval_tail,
                    sample_rate,
                ),
            }
            if val_records:
                cycle_report["val"] = evaluate_split(
                    "VAL",
                    trainer,
                    models,
                    val_records,
                    cached_waveforms,
                    frame_length,
                    frame_hop,
                    frames_per_clip,
                    drop_eval_tail,
                    sample_rate,
                )
            if args.eval_modes:
                cycle_report["test_modes"] = evaluate_split_modes(
                    "TEST",
                    models,
                    test_records,
                    cached_waveforms,
                    frame_length,
                    frame_hop,
                    frames_per_clip,
                    device,
                    args.zero_threshold,
                    drop_eval_tail,
                    sample_rate,
                    input_transform,
                )
            report["cycles"].append(cycle_report)

        output_path = os.path.join(args.exp_dir, "analysis_all_cycles.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nAnalysis written: {output_path}")
        return

    if args.ensemble_last2:
        checkpoint_paths = [
            os.path.join(args.exp_dir, "checkpoints", f"tcam_fold_{args.fold}_cycle_3.pt"),
            os.path.join(args.exp_dir, "checkpoints", f"tcam_fold_{args.fold}_cycle_4.pt"),
        ]
        models = [load_model(path, device, cfg, metrics) for path in checkpoint_paths]
        model_name = "ensemble_last2"
    else:
        checkpoint_path = args.checkpoint or os.path.join(
            args.exp_dir,
            "checkpoints",
            f"tcam_fold_{args.fold}_cycle_4.pt",
        )
        models = [load_model(checkpoint_path, device, cfg, metrics)]
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
        frame_length,
        frame_hop,
        frames_per_clip,
        drop_eval_tail,
        sample_rate,
    )
    if val_records:
        report["val"] = evaluate_split(
            "VAL",
            trainer,
            models,
            val_records,
            cached_waveforms,
            frame_length,
            frame_hop,
            frames_per_clip,
            drop_eval_tail,
            sample_rate,
        )
    if args.eval_train:
        report["train"] = evaluate_split(
            "TRAIN",
            trainer,
            models,
            train_records,
            cached_waveforms,
            frame_length,
            frame_hop,
            frames_per_clip,
            drop_eval_tail,
            sample_rate,
        )
    if args.eval_modes:
        report["test_modes"] = evaluate_split_modes(
            "TEST",
            models,
            test_records,
            cached_waveforms,
            frame_length,
            frame_hop,
            frames_per_clip,
            device,
            args.zero_threshold,
            drop_eval_tail,
            sample_rate,
            input_transform,
        )
        if args.eval_train:
            report["train_modes"] = evaluate_split_modes(
                "TRAIN",
                models,
                train_records,
                cached_waveforms,
                frame_length,
                frame_hop,
                frames_per_clip,
                device,
                args.zero_threshold,
                drop_eval_tail,
                sample_rate,
                input_transform,
            )

    output_path = os.path.join(args.exp_dir, f"analysis_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAnalysis written: {output_path}")


if __name__ == "__main__":
    main()
