import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# sklearn is only needed for probe classifiers — not for make_split / AST fine-tune.
# Lazy-import so SDP verify and teacher train work without scikit-learn installed.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import LogMelFeatureExtractor, load_audio_to_ram, parse_dataset
from train import (
    default_data_dir,
    make_stratified_clip_subset,
    make_stratified_random_clip_split,
    make_stratified_source_group_split,
    make_stratified_source_group_train_val_test_split,
    source_label_overlap_summary,
)


def _require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC, SVC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for feature-probe classifiers only. "
            "Install with: pip install scikit-learn"
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "MLPClassifier": MLPClassifier,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "LinearSVC": LinearSVC,
        "SVC": SVC,
    }


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


def pct(value):
    if value is None:
        return None
    return 100.0 * float(value)


def fmt_pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.2f}%"


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def make_split(clip_records, fold, protocol, seed):
    protocol = protocol.lower()
    uses_validation = False
    val_bucket = None
    val_clips = []

    if protocol == "paper_9_1":
        test_records = [r for r in clip_records if r["fold"] == fold]
        train_clips = [r for r in clip_records if r["fold"] != fold]
    elif protocol == "clean_8_1_1":
        test_records = [r for r in clip_records if r["fold"] == fold]
        val_bucket = (fold % 10) + 1
        train_clips = [r for r in clip_records if r["fold"] != fold and r["fold"] != val_bucket]
        val_clips = [r for r in clip_records if r["fold"] == val_bucket]
        uses_validation = True
    elif protocol == "random_clip_9_1":
        train_clips, test_records = make_stratified_random_clip_split(clip_records, fold, seed=seed)
    elif protocol == "source_group_9_1":
        train_clips, test_records = make_stratified_source_group_split(clip_records, fold, seed=seed)
    elif protocol == "source_group_8_1_1":
        train_clips, val_clips, test_records, val_bucket = make_stratified_source_group_train_val_test_split(
            clip_records,
            test_bucket=fold,
            seed=seed,
        )
        uses_validation = True
    else:
        raise ValueError(
            f"Unsupported protocol '{protocol}'. Use paper_9_1, clean_8_1_1, "
            "random_clip_9_1, source_group_9_1, or source_group_8_1_1."
        )

    return {
        "fold": fold,
        "protocol": protocol,
        "train": train_clips,
        "val": val_clips,
        "test": test_records,
        "uses_validation": uses_validation,
        "val_bucket": val_bucket,
    }


def apply_smoke_subsets(split, args):
    if args.max_train_clips is None and args.max_val_clips is None and args.max_test_clips is None:
        return split
    seed = int(args.seed)
    split = dict(split)
    split["original_counts"] = {
        "train": len(split["train"]),
        "val": len(split["val"]),
        "test": len(split["test"]),
    }
    split["train"] = make_stratified_clip_subset(split["train"], args.max_train_clips, seed + 101)
    split["val"] = make_stratified_clip_subset(split["val"], args.max_val_clips, seed + 202)
    split["test"] = make_stratified_clip_subset(split["test"], args.max_test_clips, seed + 303)
    return split


def records_to_matrix(records, feature_by_path):
    x = np.stack([feature_by_path[r["path"]] for r in records]).astype(np.float32)
    y = np.array([int(r["label"]) for r in records], dtype=np.int64)
    return x, y


def zcr_features(waveforms):
    signs = torch.signbit(waveforms[:, 0, :])
    return (signs[:, 1:] != signs[:, :-1]).float().mean(dim=1, keepdim=True)


def segment_rms_features(waveforms, segments):
    batch, _, length = waveforms.shape
    usable = (length // segments) * segments
    if usable <= 0:
        return torch.zeros(batch, segments, device=waveforms.device)
    x = waveforms[:, :, :usable].reshape(batch, 1, segments, usable // segments)
    return torch.sqrt(torch.mean(x * x, dim=(1, 3)).clamp_min(1e-12))


def band_pool_features(mel, bands=8):
    n_mels = mel.shape[1]
    band_edges = torch.linspace(0, n_mels, steps=bands + 1, device=mel.device).round().long()
    pooled = []
    for idx in range(bands):
        start = int(band_edges[idx].item())
        end = max(start + 1, int(band_edges[idx + 1].item()))
        band = mel[:, start:end, :]
        pooled.append(band.mean(dim=(1, 2), keepdim=False).unsqueeze(1))
        pooled.append(band.std(dim=(1, 2), keepdim=False).unsqueeze(1))
    return torch.cat(pooled, dim=1)


def extract_batch_features(waveforms, extractor, segments, include_delta=True):
    waveforms = waveforms.float()
    mel = extractor(waveforms)
    mel_mean = mel.mean(dim=2)
    mel_std = mel.std(dim=2)
    mel_max = mel.amax(dim=2)
    mel_min = mel.amin(dim=2)

    parts = [mel_mean, mel_std, mel_max, mel_min, band_pool_features(mel)]
    if include_delta and mel.shape[-1] > 2:
        delta = mel[:, :, 1:] - mel[:, :, :-1]
        parts.extend([delta.mean(dim=2), delta.std(dim=2), delta.amax(dim=2), delta.amin(dim=2)])

    raw = waveforms[:, 0, :]
    raw_stats = torch.stack(
        [
            raw.mean(dim=1),
            raw.std(dim=1),
            raw.abs().mean(dim=1),
            raw.abs().amax(dim=1),
            torch.sqrt(torch.mean(raw * raw, dim=1).clamp_min(1e-12)),
        ],
        dim=1,
    )
    parts.append(raw_stats)
    parts.append(zcr_features(waveforms))
    parts.append(segment_rms_features(waveforms, segments))
    return torch.cat(parts, dim=1)


def cache_key(paths, args):
    payload = {
        "paths": paths,
        "sample_rate": args.sample_rate,
        "clip_seconds": args.clip_seconds,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "win_length": args.win_length,
        "mel_hop_length": args.mel_hop_length,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "segments": args.segments,
        "include_delta": args.include_delta,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def load_waveform_batch(paths, sample_rate, load_workers):
    if load_workers <= 1:
        loaded = [load_audio_to_ram(path, sample_rate)[1] for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=load_workers) as executor:
            loaded = [wave for _, wave, _real in executor.map(lambda p: load_audio_to_ram(p, sample_rate), paths)]
    return torch.from_numpy(np.stack(loaded, axis=0))


def build_feature_cache(selected_records, args, exp_dir):
    paths = sorted({r["path"] for r in selected_records})
    key = cache_key(paths, args)
    cache_path = exp_dir / f"feature_cache_{key}.npz"
    if cache_path.exists() and not args.rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = [str(p) for p in data["paths"]]
        if cached_paths == paths:
            print(f"[Feature Cache] loaded {len(paths)} clips from {cache_path}")
            return {path: feature for path, feature in zip(cached_paths, data["features"])}
        print("[Feature Cache] cache path exists but path list changed; rebuilding.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Feature Setup] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    extractor = LogMelFeatureExtractor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.mel_hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        normalize=args.logmel_normalize,
    ).to(device).eval()

    features = []
    start_time = time.time()
    print(
        "[Feature Setup] "
        f"clips={len(paths)} | device={device} | n_mels={args.n_mels} | "
        f"hop={args.mel_hop_length} | batch_size={args.feature_batch_size}"
    )
    with torch.no_grad():
        for start in range(0, len(paths), args.feature_batch_size):
            batch_paths = paths[start:start + args.feature_batch_size]
            batch = load_waveform_batch(batch_paths, args.sample_rate, args.load_workers).to(device)
            batch_features = extract_batch_features(
                batch,
                extractor,
                segments=args.segments,
                include_delta=args.include_delta,
            )
            features.append(batch_features.cpu().numpy().astype(np.float32))
            done = min(start + args.feature_batch_size, len(paths))
            if done == len(paths) or done % max(args.feature_batch_size * 10, 1) == 0:
                elapsed = time.time() - start_time
                print(f"[Feature Setup] processed {done}/{len(paths)} clips in {elapsed:.1f}s")

    matrix = np.concatenate(features, axis=0)
    np.savez_compressed(cache_path, paths=np.array(paths, dtype=object), features=matrix)
    print(f"[Feature Cache] saved {matrix.shape[0]}x{matrix.shape[1]} features to {cache_path}")
    return {path: feature for path, feature in zip(paths, matrix)}


def build_estimator(name, seed, n_jobs, n_estimators):
    sk = _require_sklearn()
    make_pipeline = sk["make_pipeline"]
    StandardScaler = sk["StandardScaler"]
    LogisticRegression = sk["LogisticRegression"]
    LinearSVC = sk["LinearSVC"]
    SVC = sk["SVC"]
    ExtraTreesClassifier = sk["ExtraTreesClassifier"]
    RandomForestClassifier = sk["RandomForestClassifier"]
    MLPClassifier = sk["MLPClassifier"]

    name = name.lower()
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=3.0,
                max_iter=2500,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.8,
                class_weight="balanced",
                max_iter=10000,
                random_state=seed,
            ),
        )
    if name == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=8.0,
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(512, 256),
                activation="relu",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=350,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unsupported model '{name}'.")


def per_class_rows(y_true, y_pred):
    confusion_matrix = _require_sklearn()["confusion_matrix"]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        support = int(matrix[class_id].sum())
        correct = int(matrix[class_id, class_id])
        predicted_as = int(matrix[:, class_id].sum())
        confusions = []
        for pred_id, count in enumerate(matrix[class_id]):
            if pred_id != class_id and count > 0:
                confusions.append({"predicted": CLASS_NAMES[pred_id], "count": int(count)})
        confusions.sort(key=lambda item: item["count"], reverse=True)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "correct": correct,
                "accuracy": correct / support if support else None,
                "predicted_count": predicted_as,
                "top_confusions": confusions[:4],
            }
        )
    return rows


def weak_source_groups(test_records, predictions, min_support=5, limit=8):
    by_group = collections.defaultdict(lambda: {"support": 0, "correct": 0, "predictions": collections.Counter()})
    for record, predicted in zip(test_records, predictions):
        key = (int(record["label"]), str(record.get("fsID", "")), int(record.get("classID", record["label"])))
        item = by_group[key]
        item["support"] += 1
        item["correct"] += int(int(predicted) == int(record["label"]))
        item["predictions"][int(predicted)] += 1

    rows = []
    for (label, fsid, class_id), item in by_group.items():
        support = item["support"]
        if support < min_support:
            continue
        correct = item["correct"]
        wrong = []
        for pred_id, count in item["predictions"].most_common():
            if pred_id != label:
                wrong.append({"predicted": CLASS_NAMES[pred_id], "count": int(count)})
        rows.append(
            {
                "class_id": label,
                "class_name": CLASS_NAMES[label],
                "fsID": fsid,
                "metadata_classID": class_id,
                "support": support,
                "correct": correct,
                "accuracy": correct / support if support else None,
                "top_wrong": wrong[:4],
            }
        )
    rows.sort(key=lambda row: (row["accuracy"], -row["support"], row["class_name"], row["fsID"]))
    return rows[:limit]


def evaluate_estimator(name, estimator, split, feature_by_path, args):
    x_train, y_train = records_to_matrix(split["train"], feature_by_path)
    x_test, y_test = records_to_matrix(split["test"], feature_by_path)
    x_val = y_val = None
    if split["uses_validation"] and split["val"]:
        x_val, y_val = records_to_matrix(split["val"], feature_by_path)

    start_time = time.time()
    estimator.fit(x_train, y_train)
    fit_seconds = time.time() - start_time

    test_pred = estimator.predict(x_test)
    accuracy_score = _require_sklearn()["accuracy_score"]
    test_acc = accuracy_score(y_test, test_pred)
    val_acc = None
    if x_val is not None:
        val_pred = estimator.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred)

    class_rows = per_class_rows(y_test, test_pred)
    worst = min(
        (row for row in class_rows if row["accuracy"] is not None),
        key=lambda row: row["accuracy"],
    )
    return {
        "model": name,
        "fit_seconds": fit_seconds,
        "train_count": int(len(y_train)),
        "val_count": int(len(y_val)) if y_val is not None else 0,
        "test_count": int(len(y_test)),
        "val_acc": val_acc,
        "test_acc": test_acc,
        "worst_class": worst["class_name"],
        "worst_class_acc": worst["accuracy"],
        "per_class": class_rows,
        "weak_source_groups": weak_source_groups(
            split["test"],
            test_pred,
            min_support=args.min_source_support,
            limit=args.weak_group_limit,
        ),
    }


def aggregate_results(fold_results, model_names):
    aggregate = {}
    for model_name in model_names:
        rows = []
        for fold in fold_results:
            item = fold["models"].get(model_name)
            if item is not None:
                rows.append(item)
        val_mean, val_std = mean_std([pct(row["val_acc"]) for row in rows])
        test_mean, test_std = mean_std([pct(row["test_acc"]) for row in rows])
        worst_mean, worst_std = mean_std([pct(row["worst_class_acc"]) for row in rows])
        aggregate[model_name] = {
            "folds": len(rows),
            "val_acc_pct": {"mean": val_mean, "std": val_std},
            "test_acc_pct": {"mean": test_mean, "std": test_std},
            "worst_class_acc_pct": {"mean": worst_mean, "std": worst_std},
        }

    selected_tests = []
    selected_vals = []
    oracle_tests = []
    selected_per_class = collections.defaultdict(list)
    for fold in fold_results:
        model_rows = list(fold["models"].values())
        valid_val = [row for row in model_rows if row["val_acc"] is not None]
        if valid_val:
            selected = max(valid_val, key=lambda row: row["val_acc"])
        else:
            selected = max(model_rows, key=lambda row: row["test_acc"])
        oracle = max(model_rows, key=lambda row: row["test_acc"])
        fold["selected_by_val"] = {
            "model": selected["model"],
            "val_acc": selected["val_acc"],
            "test_acc": selected["test_acc"],
        }
        fold["oracle_best_test"] = {
            "model": oracle["model"],
            "test_acc": oracle["test_acc"],
        }
        selected_vals.append(pct(selected["val_acc"]))
        selected_tests.append(pct(selected["test_acc"]))
        oracle_tests.append(pct(oracle["test_acc"]))
        for item in selected["per_class"]:
            selected_per_class[item["class_name"]].append(pct(item["accuracy"]))

    selected_test_mean, selected_test_std = mean_std(selected_tests)
    selected_val_mean, selected_val_std = mean_std(selected_vals)
    oracle_test_mean, oracle_test_std = mean_std(oracle_tests)
    per_class = {}
    for class_name in CLASS_NAMES:
        mean, std = mean_std(selected_per_class[class_name])
        per_class[class_name] = {"mean": mean, "std": std}

    return {
        "by_model": aggregate,
        "selected_by_val": {
            "val_acc_pct": {"mean": selected_val_mean, "std": selected_val_std},
            "test_acc_pct": {"mean": selected_test_mean, "std": selected_test_std},
        },
        "oracle_best_test": {
            "test_acc_pct": {"mean": oracle_test_mean, "std": oracle_test_std},
            "note": "Diagnostic only. This selects by test accuracy and is not a valid model-selection result.",
        },
        "selected_by_val_per_class_test_pct": per_class,
    }


def write_summary(summary, exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)
    json_path = exp_dir / "feature_probe_summary.json"
    md_path = exp_dir / "feature_probe_summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    lines = [
        f"# Source-Safe Feature Probe: {exp_dir.name}",
        "",
        "This is not the deployable 1D-CNN. It is a fast upper-bound/probe to test whether stronger clip-level features separate the current leakage-safe split.",
        "",
        "## Fold Results",
        "",
        "| Fold | Model | Val | Test | Worst class | Worst acc | Fit seconds |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for fold in summary["fold_results"]:
        for model_name, row in fold["models"].items():
            lines.append(
                "| "
                f"{fold['fold']} | "
                f"{model_name} | "
                f"{fmt_pct(pct(row['val_acc']))} | "
                f"{fmt_pct(pct(row['test_acc']))} | "
                f"{row['worst_class']} | "
                f"{fmt_pct(pct(row['worst_class_acc']))} | "
                f"{row['fit_seconds']:.1f} |"
            )
    lines.extend(["", "## Validation-Selected Result", ""])
    lines.extend(
        [
            "| Fold | Selected model | Val | Test | Oracle best-test model | Oracle test |",
            "|---:|---|---:|---:|---|---:|",
        ]
    )
    for fold in summary["fold_results"]:
        selected = fold["selected_by_val"]
        oracle = fold["oracle_best_test"]
        lines.append(
            "| "
            f"{fold['fold']} | "
            f"{selected['model']} | "
            f"{fmt_pct(pct(selected['val_acc']))} | "
            f"{fmt_pct(pct(selected['test_acc']))} | "
            f"{oracle['model']} | "
            f"{fmt_pct(pct(oracle['test_acc']))} |"
        )
    selected = summary["aggregate"]["selected_by_val"]
    oracle = summary["aggregate"]["oracle_best_test"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Result | Mean | Std |",
            "|---|---:|---:|",
            f"| Selected-by-val validation | {fmt_pct(selected['val_acc_pct']['mean'])} | {fmt_pct(selected['val_acc_pct']['std'])} |",
            f"| Selected-by-val test | {fmt_pct(selected['test_acc_pct']['mean'])} | {fmt_pct(selected['test_acc_pct']['std'])} |",
            f"| Oracle best-test diagnostic | {fmt_pct(oracle['test_acc_pct']['mean'])} | {fmt_pct(oracle['test_acc_pct']['std'])} |",
            "",
            "## Aggregate By Model",
            "",
            "| Model | Folds | Val mean | Test mean | Worst-class mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, row in summary["aggregate"]["by_model"].items():
        lines.append(
            "| "
            f"{model_name} | "
            f"{row['folds']} | "
            f"{fmt_pct(row['val_acc_pct']['mean'])} | "
            f"{fmt_pct(row['test_acc_pct']['mean'])} | "
            f"{fmt_pct(row['worst_class_acc_pct']['mean'])} |"
        )

    lines.extend(["", "## Selected-By-Val Per Class", ""])
    lines.extend(["| Class | Test mean | Std |", "|---|---:|---:|"])
    for class_name, row in summary["aggregate"]["selected_by_val_per_class_test_pct"].items():
        lines.append(f"| {class_name} | {fmt_pct(row['mean'])} | {fmt_pct(row['std'])} |")

    lines.extend(["", "## Weak Source Groups", ""])
    lines.extend(["| Fold | Model | Class | fsID | Correct/support | Acc | Main wrong |", "|---:|---|---|---:|---:|---:|---|"])
    for fold in summary["fold_results"]:
        selected_model = fold["selected_by_val"]["model"]
        for row in fold["models"][selected_model]["weak_source_groups"][:6]:
            wrong = ", ".join(f"{item['predicted']}:{item['count']}" for item in row["top_wrong"])
            lines.append(
                "| "
                f"{fold['fold']} | "
                f"{selected_model} | "
                f"{row['class_name']} | "
                f"{row['fsID']} | "
                f"{row['correct']}/{row['support']} | "
                f"{fmt_pct(pct(row['accuracy']))} | "
                f"{wrong} |"
            )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe classical feature probe for UrbanSound8K.")
    parser.add_argument("--data_dir", default=default_data_dir())
    parser.add_argument("--exp_name", default="source_safe_feature_probe")
    parser.add_argument("--folds", default="1-3")
    parser.add_argument("--protocol", default="source_group_8_1_1")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--models", default="logreg,linear_svm,extra_trees")
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_seconds", type=float, default=4.0)
    parser.add_argument("--n_mels", type=int, default=96)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--mel_hop_length", type=int, default=256)
    parser.add_argument("--f_min", type=float, default=40.0)
    parser.add_argument("--f_max", type=float, default=8000.0)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--include_delta", action="store_true", default=True)
    parser.add_argument("--no_delta", action="store_false", dest="include_delta")
    parser.add_argument("--logmel_normalize", action="store_true", default=False)
    parser.add_argument("--feature_batch_size", type=int, default=96)
    parser.add_argument("--load_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--max_test_clips", type=int, default=None)
    parser.add_argument("--min_source_support", type=int, default=5)
    parser.add_argument("--weak_group_limit", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = REPO_ROOT / "experiments" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(args.data_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")
    clip_records = parse_dataset(csv_path, audio_base, CLASS_NAMES)
    folds = parse_folds(args.folds)
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]

    splits = []
    selected_records_by_path = {}
    for fold in folds:
        split = make_split(clip_records, fold, args.protocol, seed=args.seed)
        split = apply_smoke_subsets(split, args)
        splits.append(split)
        for record in split["train"] + split["val"] + split["test"]:
            selected_records_by_path[record["path"]] = record

        overlap = source_label_overlap_summary(split["train"], split["test"])
        val_overlap = source_label_overlap_summary(split["train"], split["val"]) if split["uses_validation"] else {"count": None}
        print(
            f"[Split] fold={fold} protocol={args.protocol} "
            f"clips train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} "
            f"train/test source-label overlap={overlap['count']} "
            f"train/val source-label overlap={val_overlap['count']}"
        )

    feature_by_path = build_feature_cache(list(selected_records_by_path.values()), args, exp_dir)
    feature_dim = len(next(iter(feature_by_path.values())))

    fold_results = []
    for split in splits:
        print(f"\n=================== FEATURE PROBE FOLD {split['fold']} ===================")
        model_results = {}
        for name in model_names:
            estimator = build_estimator(name, args.seed + split["fold"], args.n_jobs, args.n_estimators)
            result = evaluate_estimator(name, estimator, split, feature_by_path, args)
            model_results[name] = result
            print(
                f"{name:<12} | Val={fmt_pct(pct(result['val_acc']))} | "
                f"Test={fmt_pct(pct(result['test_acc']))} | "
                f"Worst={result['worst_class']} {fmt_pct(pct(result['worst_class_acc']))} | "
                f"Fit={result['fit_seconds']:.1f}s"
            )
        fold_results.append(
            {
                "fold": split["fold"],
                "protocol": args.protocol,
                "val_bucket": split["val_bucket"],
                "counts": {
                    "train": len(split["train"]),
                    "val": len(split["val"]),
                    "test": len(split["test"]),
                },
                "models": model_results,
            }
        )

    summary = {
        "exp_name": args.exp_name,
        "protocol": args.protocol,
        "seed": args.seed,
        "folds": folds,
        "models": model_names,
        "feature_dim": feature_dim,
        "feature_extractor": {
            "sample_rate": args.sample_rate,
            "clip_seconds": args.clip_seconds,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "win_length": args.win_length,
            "mel_hop_length": args.mel_hop_length,
            "f_min": args.f_min,
            "f_max": args.f_max,
            "segments": args.segments,
            "include_delta": args.include_delta,
            "logmel_normalize": args.logmel_normalize,
        },
        "fold_results": fold_results,
    }
    summary["aggregate"] = aggregate_results(fold_results, model_names)
    json_path, md_path = write_summary(summary, exp_dir)
    print(f"\nSummary written: {json_path}")
    print(f"Summary report : {md_path}")


if __name__ == "__main__":
    main()
