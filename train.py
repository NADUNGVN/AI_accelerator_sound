import os
import sys
import argparse
import copy
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Ensure local src directory is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import (
    TCAMAttn1DNet,
    TCAM1DCNN,
    DSRes1DSENet,
    EfficientAudioCNN1D,
    DSConv2DH1PyramidNet,
    DSConv2DH1PyramidNetDeep,
    DSConv2DH1LogMelNet,
    KV260AudioNetDS1D,
    KV260AudioNetDS1DDeep,
    KV260LogMelNetDS1D,
)
from src.data import (
    CachedUrbanSoundFrameDataset,
    LogMelFeatureExtractor,
    parse_audio_dataset,
    normalize_dataset_name,
    generate_frame_records,
    load_audio_to_ram,
)
from src.training import Trainer
from src.utils import set_seed, prepare_dirs


RANDOM_SPLIT_ALGORITHM = "stable_metadata_v2"
SOURCE_GROUP_SPLIT_ALGORITHM = "fsid_classid_balanced_v1"


def default_data_dir(dataset_name="urbansound8k"):
    """
    Prefer the repo-local layout, but also support the shared research dataset
    layout used by this workspace.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if dataset_name == "urbansound8k":
        candidates = [
            os.path.join(repo_root, "data", "raw", "UrbanSound8K"),
            os.path.abspath(os.path.join(repo_root, "..", "..", "..", "data", "UrbanSound8K")),
        ]
        marker = os.path.join("metadata", "UrbanSound8K.csv")
    elif dataset_name in {"esc50", "esc10"}:
        candidates = [
            os.path.join(repo_root, "data", "raw", "ESC-50"),
            os.path.abspath(os.path.join(repo_root, "..", "..", "..", "data", "ESC-50")),
            os.path.abspath(os.path.join(repo_root, "..", "..", "..", "data", "esc50")),
        ]
        marker = os.path.join("meta", "esc50.csv")
    elif dataset_name == "speech_commands":
        candidates = [
            os.path.join(repo_root, "data", "raw", "speech_commands_v0.02"),
            os.path.join(repo_root, "data", "raw", "SpeechCommands"),
            os.path.abspath(os.path.join(repo_root, "..", "..", "..", "data", "speech_commands_v0.02")),
            os.path.abspath(os.path.join(repo_root, "..", "..", "..", "data", "SpeechCommands")),
        ]
        marker = "validation_list.txt"
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, marker)):
            return candidate
    return candidates[0]


def random_split_sort_key(record):
    return (
        record["label"],
        record["fold"],
        record.get("slice_file_name", os.path.basename(record["path"])),
        str(record.get("fsID", "")),
        int(record.get("classID", -1)),
    )


def make_stratified_clip_subset(records, max_clips, seed):
    """
    Deterministically limits a split for smoke tests while keeping class coverage
    roughly balanced. Sampling happens after the real split, so it cannot create
    leakage that was not already present.
    """
    if max_clips is None or max_clips >= len(records):
        return records
    if max_clips <= 0:
        raise ValueError(f"max_clips must be positive when provided, got {max_clips}")

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    for label in by_label:
        by_label[label] = sorted(by_label[label], key=random_split_sort_key)
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


def make_stratified_random_clip_split(clip_records, test_bucket, seed, num_buckets=10):
    """
    Creates a reproducible stratified random clip split.
    Frames are generated only after this split, so frames from the same clip
    cannot appear in both train and test. Official source-level grouping is not
    preserved; source overlap is reported separately for interpretation.
    """
    if not 1 <= test_bucket <= num_buckets:
        raise ValueError(f"test_bucket must be in [1, {num_buckets}], got {test_bucket}")

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for record in clip_records:
        by_class[record["label"]].append(record)

    train_clips = []
    test_records = []
    for label in sorted(by_class):
        records = sorted(by_class[label], key=random_split_sort_key)
        rng.shuffle(records)
        for idx, record in enumerate(records):
            bucket = (idx % num_buckets) + 1
            if bucket == test_bucket:
                test_records.append(record)
            else:
                train_clips.append(record)

    return train_clips, test_records


def make_stratified_source_group_split(clip_records, test_bucket, seed, num_buckets=10):
    """
    Creates a reproducible stratified random split at source-label group level.
    All clips sharing the same (fsID, classID) stay on the same side of the
    train/test boundary, preventing the source-label leakage seen in random
    clip-level splitting.
    """
    if not 1 <= test_bucket <= num_buckets:
        raise ValueError(f"test_bucket must be in [1, {num_buckets}], got {test_bucket}")

    if not clip_records or "fsID" not in clip_records[0] or "classID" not in clip_records[0]:
        raise ValueError("source_group_9_1 requires fsID and classID metadata fields.")

    buckets_by_class = make_stratified_source_group_buckets(clip_records, seed, num_buckets=num_buckets)
    train_clips = []
    test_records = []
    for label_buckets in buckets_by_class.values():
        for idx, records in enumerate(label_buckets, start=1):
            if idx == test_bucket:
                test_records.extend(records)
            else:
                train_clips.extend(records)

    return train_clips, test_records


def make_stratified_source_group_buckets(clip_records, seed, num_buckets=10):
    if not clip_records or "fsID" not in clip_records[0] or "classID" not in clip_records[0]:
        raise ValueError("source-group split requires fsID and classID metadata fields.")

    rng = random.Random(seed)
    groups_by_class = defaultdict(dict)
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
    if not 1 <= test_bucket <= num_buckets:
        raise ValueError(f"test_bucket must be in [1, {num_buckets}], got {test_bucket}")

    val_bucket = (test_bucket % num_buckets) + 1
    buckets_by_class = make_stratified_source_group_buckets(clip_records, seed, num_buckets=num_buckets)

    train_clips = []
    val_clips = []
    test_records = []
    for label_buckets in buckets_by_class.values():
        for idx, records in enumerate(label_buckets, start=1):
            if idx == test_bucket:
                test_records.extend(records)
            elif idx == val_bucket:
                val_clips.extend(records)
            else:
                train_clips.extend(records)

    return train_clips, val_clips, test_records, val_bucket


def source_label_overlap_summary(train_clips, test_records, limit=10):
    if not train_clips or not test_records:
        return {"count": 0, "examples": []}
    if "fsID" not in train_clips[0] or "classID" not in train_clips[0]:
        return {"count": None, "examples": []}

    train_keys = {(r["fsID"], r["classID"]) for r in train_clips}
    test_keys = {(r["fsID"], r["classID"]) for r in test_records}
    overlap = sorted(train_keys & test_keys)
    return {
        "count": len(overlap),
        "examples": [{"fsID": fsid, "classID": class_id} for fsid, class_id in overlap[:limit]],
    }


def sorted_unique(values):
    return sorted(set(values), key=lambda value: str(value))


def validate_fold_for_protocol(protocol, fold):
    if protocol in {
        "esc50_3_1_1_foldk_valnext_v1",
        "esc50_official_4_1_cv",
        "esc10_3_1_1_foldk_valnext_v1",
        "esc10_official_4_1_cv",
    }:
        if not 1 <= fold <= 5:
            raise ValueError(f"--fold must be in [1, 5] for ESC protocols, got {fold}")
    elif protocol == "speech_commands_v2_official12":
        if fold != 1:
            raise ValueError("Speech Commands official split is not fold-based; use --fold 1.")
    else:
        if not 1 <= fold <= 10:
            raise ValueError(f"--fold must be in [1, 10], got {fold}")


def build_dataset_split(clip_records, dataset_name, protocol, fold, seed):
    dataset_name = normalize_dataset_name(dataset_name)
    val_fold = None
    val_clips = []
    uses_validation = False

    if protocol == "paper_9_1":
        test_records = [r for r in clip_records if r["fold"] == fold]
        train_clips = [r for r in clip_records if r["fold"] != fold]
        description = "paper_9_1 | Train=9 folds, Test=1 fold, no validation-based model selection."
    elif protocol == "clean_8_1_1":
        test_records = [r for r in clip_records if r["fold"] == fold]
        val_fold = (fold % 10) + 1
        train_clips = [r for r in clip_records if r["fold"] != fold and r["fold"] != val_fold]
        val_clips = [r for r in clip_records if r["fold"] == val_fold]
        uses_validation = True
        description = f"clean_8_1_1 | Train=8 folds, Val=fold {val_fold}, Test=fold {fold}."
    elif protocol == "random_clip_9_1":
        train_clips, test_records = make_stratified_random_clip_split(
            clip_records,
            test_bucket=fold,
            seed=seed,
        )
        description = (
            "random_clip_9_1 | Stratified random clip-level 9/1 control, "
            f"Test bucket={fold}, seed={seed}, split_algorithm={RANDOM_SPLIT_ALGORITHM}."
        )
    elif protocol == "source_group_9_1":
        train_clips, test_records = make_stratified_source_group_split(
            clip_records,
            test_bucket=fold,
            seed=seed,
        )
        description = (
            "source_group_9_1 | Stratified random source-label-group 9/1 control, "
            f"Test bucket={fold}, seed={seed}, split_algorithm={SOURCE_GROUP_SPLIT_ALGORITHM}."
        )
    elif protocol == "source_group_8_1_1":
        train_clips, val_clips, test_records, val_fold = make_stratified_source_group_train_val_test_split(
            clip_records,
            test_bucket=fold,
            seed=seed,
        )
        uses_validation = True
        description = (
            "source_group_8_1_1 | Stratified random source-label-group split, "
            f"Train=8 buckets, Val=bucket {val_fold}, Test=bucket {fold}, "
            f"seed={seed}, split_algorithm={SOURCE_GROUP_SPLIT_ALGORITHM}."
        )
    elif protocol in {"esc50_3_1_1_foldk_valnext_v1", "esc10_3_1_1_foldk_valnext_v1"}:
        required_dataset = "esc10" if protocol.startswith("esc10") else "esc50"
        if dataset_name != required_dataset:
            raise ValueError(f"Protocol {protocol} requires dataset='{required_dataset}'.")
        val_fold = (fold % 5) + 1
        test_records = [r for r in clip_records if r["fold"] == fold]
        val_clips = [r for r in clip_records if r["fold"] == val_fold]
        train_clips = [r for r in clip_records if r["fold"] not in {fold, val_fold}]
        uses_validation = True
        description = (
            f"{required_dataset}_3_1_1 | Train=3 folds, Val=fold {val_fold}, Test=fold {fold}."
        )
    elif protocol in {"esc50_official_4_1_cv", "esc10_official_4_1_cv"}:
        required_dataset = "esc10" if protocol.startswith("esc10") else "esc50"
        if dataset_name != required_dataset:
            raise ValueError(f"Protocol {protocol} requires dataset='{required_dataset}'.")
        test_records = [r for r in clip_records if r["fold"] == fold]
        train_clips = [r for r in clip_records if r["fold"] != fold]
        description = f"{required_dataset}_official_4_1_cv | Train=4 folds, Test=fold {fold}, no validation."
    elif protocol == "speech_commands_v2_official12":
        if dataset_name != "speech_commands":
            raise ValueError(f"Protocol {protocol} requires dataset='speech_commands'.")
        train_clips = [r for r in clip_records if r.get("split") == "train"]
        val_clips = [r for r in clip_records if r.get("split") == "validation"]
        test_records = [r for r in clip_records if r.get("split") == "test"]
        val_fold = "validation"
        uses_validation = True
        description = "speech_commands_v2_official12 | Official train/validation/test split."
    else:
        raise ValueError(f"Unsupported protocol '{protocol}'.")

    if not train_clips:
        raise RuntimeError(f"Protocol {protocol} produced an empty train split.")
    if not test_records:
        raise RuntimeError(f"Protocol {protocol} produced an empty test split.")
    if uses_validation and not val_clips:
        raise RuntimeError(f"Protocol {protocol} requires validation but produced an empty val split.")

    return train_clips, val_clips, test_records, val_fold, uses_validation, description


def build_model(cfg, num_classes):
    """Build model from config. Prefer paper names; legacy keys remain valid."""
    model_name = cfg.get("model_name", "ds_conv2d_h1_pyramid").lower()

    # Canonical paper names (preferred) and legacy aliases.
    if model_name in {"tcam_attn1d", "tcam1dcnn"}:
        model = TCAMAttn1DNet(num_classes=num_classes)
    elif model_name in {"ds_res1d_se", "efficient_audio_cnn1d"}:
        model = DSRes1DSENet(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.25)),
        )
    elif model_name in {"ds_conv2d_h1_pyramid", "kv260_audio_net_ds1d"}:
        model = DSConv2DH1PyramidNet(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.15)),
            pool_type=cfg.get("pool_type", "avg"),
            pool_bins=cfg.get("pool_bins", None),
            stem_type=cfg.get("stem_type", "single"),
            extra_late_blocks=int(cfg.get("extra_late_blocks", 0)),
        )
    elif model_name in {"ds_conv2d_h1_pyramid_deep", "kv260_audio_net_ds1d_deep"}:
        model = DSConv2DH1PyramidNetDeep(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.20)),
            pool_type=cfg.get("pool_type", "avg"),
            pool_bins=cfg.get("pool_bins", None),
            stem_type=cfg.get("stem_type", "single"),
        )
    elif model_name in {"ds_conv2d_h1_logmel", "kv260_logmel_net_ds1d"}:
        model = DSConv2DH1LogMelNet(
            num_classes=num_classes,
            input_channels=int(cfg.get("n_mels", 64)),
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.20)),
            pool_type=cfg.get("pool_type", "avgmax"),
        )
    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Prefer paper names: "
            "'ds_conv2d_h1_pyramid', 'ds_res1d_se', 'tcam_attn1d', "
            "'ds_conv2d_h1_pyramid_deep', 'ds_conv2d_h1_logmel'. "
            "Legacy: 'kv260_audio_net_ds1d', 'efficient_audio_cnn1d', 'tcam1dcnn'."
        )
    return model_name, model


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
        ).to(device)
    raise ValueError(f"Unsupported input_features '{input_features}'. Use 'waveform' or 'logmel'.")


def get_epoch_lr(epoch, total_epochs, base_lr, cycles, cfg):
    schedule_cfg = cfg.get("lr_schedule", {}) or {}
    schedule_type = str(schedule_cfg.get("type", "cosine_restart")).lower()
    base_lr = float(base_lr)

    if schedule_type in {"cosine", "cosine_restart"}:
        lr = Trainer.get_cosine_lr(epoch, total_epochs, base_lr, cycles)
    elif schedule_type in {"step", "multistep"}:
        lr = base_lr
        gamma = float(schedule_cfg.get("gamma", 0.1))
        milestones = sorted(int(m) for m in schedule_cfg.get("milestones", []))
        completed_epoch = epoch + 1
        for milestone in milestones:
            if completed_epoch > milestone:
                lr *= gamma
    elif schedule_type == "constant":
        lr = base_lr
    else:
        raise ValueError(
            f"Unsupported lr_schedule type '{schedule_type}'. "
            "Use 'cosine_restart', 'multistep', 'step', or 'constant'."
        )

    warmup_epochs = int(schedule_cfg.get("warmup_epochs", 0))
    if warmup_epochs > 0 and (epoch + 1) <= warmup_epochs:
        lr *= float(schedule_cfg.get("warmup_factor", 0.1))

    return max(lr, float(schedule_cfg.get("min_lr", 1e-6)))


def describe_lr_schedule(cfg, cycles):
    schedule_cfg = cfg.get("lr_schedule", {}) or {}
    schedule_type = str(schedule_cfg.get("type", "cosine_restart")).lower()
    parts = [f"type={schedule_type}"]
    if schedule_type in {"cosine", "cosine_restart"}:
        parts.append(f"cycles={cycles}")
    if schedule_type in {"step", "multistep"}:
        parts.append(f"milestones={schedule_cfg.get('milestones', [])}")
        parts.append(f"gamma={float(schedule_cfg.get('gamma', 0.1)):g}")
    if int(schedule_cfg.get("warmup_epochs", 0)) > 0:
        parts.append(f"warmup_epochs={int(schedule_cfg.get('warmup_epochs', 0))}")
        parts.append(f"warmup_factor={float(schedule_cfg.get('warmup_factor', 0.1)):g}")
    parts.append(f"min_lr={float(schedule_cfg.get('min_lr', 1e-6)):g}")
    return " | ".join(parts)


def resolve_repo_path(path):
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(repo_root, path))


def load_json_config(path):
    resolved = resolve_repo_path(path)
    with open(resolved, "r") as f:
        return json.load(f), resolved


def format_teacher_checkpoint_path(template, fold):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    formatted = str(template).format(fold=fold, repo_root=repo_root)
    return resolve_repo_path(formatted)


def load_teacher_model(student_cfg, distillation_cfg, fold, device, num_classes):
    if not distillation_cfg.get("enabled", False):
        return None, None, None, None

    teacher_config_path = distillation_cfg.get("teacher_config")
    if teacher_config_path:
        teacher_cfg, teacher_config_resolved = load_json_config(teacher_config_path)
    else:
        teacher_cfg = copy.deepcopy(student_cfg)
        teacher_config_resolved = None

    compatibility_keys = ["input_features", "sample_rate", "frame_length", "frame_hop", "frames_per_clip"]
    mismatches = []
    for key in compatibility_keys:
        student_value = student_cfg.get(key)
        teacher_value = teacher_cfg.get(key)
        if student_value != teacher_value:
            mismatches.append(f"{key}: student={student_value} teacher={teacher_value}")
    if mismatches:
        raise ValueError(
            "Distillation teacher/student input settings must match for this trainer: "
            + "; ".join(mismatches)
        )

    checkpoint_template = distillation_cfg.get("teacher_checkpoint_template") or distillation_cfg.get("checkpoint_template")
    if not checkpoint_template:
        raise ValueError("distillation.teacher_checkpoint_template is required when distillation is enabled.")
    checkpoint_path = format_teacher_checkpoint_path(checkpoint_template, fold)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    teacher_name, teacher_model = build_model(teacher_cfg, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    teacher_model.load_state_dict(state)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    return teacher_name, teacher_model, checkpoint_path, teacher_config_resolved


def load_initial_model_weights(model, cfg, fold, device):
    initial_cfg = cfg.get("initial_checkpoint", {}) or {}
    if not initial_cfg.get("enabled", False):
        return None

    checkpoint_template = initial_cfg.get("template") or initial_cfg.get("path")
    if not checkpoint_template:
        raise ValueError("initial_checkpoint.template is required when initial_checkpoint is enabled.")
    checkpoint_path = format_teacher_checkpoint_path(checkpoint_template, fold)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    strict = bool(initial_cfg.get("strict", True))
    model.load_state_dict(state, strict=strict)
    print(
        "[Initial Checkpoint Setup] enabled=True | "
        f"checkpoint={checkpoint_path} | strict={strict}"
    )
    return checkpoint_path


def count_parameters(model):
    return {
        "params_with_bias": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params_no_bias": sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.endswith("bias")
        ),
    }


def estimate_conv_linear_macs(model, input_length, device, input_channels=1):
    hooks = []
    macs = 0

    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            batch, out_channels, out_length = output.shape
            kernel = module.kernel_size[0]
            in_channels = module.in_channels // module.groups
            macs += int(batch * out_channels * out_length * in_channels * kernel)
        elif isinstance(module, nn.Conv2d):
            batch, out_channels, out_height, out_width = output.shape
            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels // module.groups
            macs += int(batch * out_channels * out_height * out_width * in_channels * kernel_h * kernel_w)
        elif isinstance(module, nn.Linear):
            batch = output.shape[0]
            macs += int(batch * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, input_channels, input_length, device=device))
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return macs


def enforce_deployment_budget(cfg, model_params, model_macs_per_clip):
    budget = cfg.get("deployment_budget", {}) or {}
    max_params = budget.get("max_params", cfg.get("max_params"))
    max_macs = budget.get("max_macs_per_clip", cfg.get("max_macs_per_clip"))
    if max_params is None and max_macs is None:
        return

    params = int(model_params["params_with_bias"])
    print(
        "[Budget Setup] "
        f"max_params={max_params if max_params is not None else 'none'} | "
        f"max_macs_per_clip={max_macs if max_macs is not None else 'none'} | "
        f"actual_params={params:,} | actual_macs_per_clip={model_macs_per_clip:,}"
    )
    if max_params is not None and params > int(max_params):
        raise ValueError(
            f"Model parameter budget exceeded: {params:,} > {int(max_params):,}. "
            "Reduce width_mult/channels or increase deployment_budget.max_params."
        )
    if max_macs is not None and model_macs_per_clip > int(max_macs):
        raise ValueError(
            f"Model MAC/clip budget exceeded: {model_macs_per_clip:,} > {int(max_macs):,}. "
            "Reduce width_mult/input length/frames_per_clip or increase deployment_budget.max_macs_per_clip."
        )


def balanced_class_weights(records, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for record in records:
        counts[int(record["label"])] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def apply_class_multipliers(weights, multipliers, device):
    if multipliers is None:
        return weights
    if len(multipliers) != weights.numel():
        raise ValueError(f"Expected {weights.numel()} class multipliers, got {len(multipliers)}")
    multiplier_tensor = torch.tensor(multipliers, dtype=torch.float32, device=device)
    return weights * multiplier_tensor


def make_weighted_sampler(frame_records, num_classes, multipliers=None):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    labels = []
    for record in frame_records:
        label = int(record["label"])
        labels.append(label)
        counts[label] += 1.0
    counts = counts.clamp_min(1.0)
    class_weights = counts.sum() / (num_classes * counts)
    if multipliers is not None:
        class_weights = apply_class_multipliers(class_weights, multipliers, device=torch.device("cpu"))
    sample_weights = torch.tensor([float(class_weights[label]) for label in labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def attach_source_ids(frame_records):
    source_to_id = {}
    for record in frame_records:
        fsid = record.get("fsID")
        if fsid is None:
            fsid = os.path.basename(record["path"]).split("-")[0]
        class_id = record.get("classID", record["label"])
        source_key = (str(fsid), int(class_id))
        if source_key not in source_to_id:
            source_to_id[source_key] = len(source_to_id)
        record["source_id"] = source_to_id[source_key]
    return source_to_id


class SourceAwareBatchSampler(Sampler):
    """
    Builds batches with multiple source groups per class. This can be used for
    source-robust CE training and also gives supervised contrastive training
    positive pairs that share the label but not the source.
    """
    def __init__(
        self,
        records,
        batch_size,
        classes_per_batch=8,
        sources_per_class=2,
        samples_per_source=4,
        class_multipliers=None,
        seed=83,
    ):
        self.records = records
        self.batch_size = max(1, int(batch_size))
        self.sources_per_class = max(1, int(sources_per_class))
        self.samples_per_source = max(1, int(samples_per_source))
        self.classes_per_batch = max(1, int(classes_per_batch))
        self.class_multipliers = class_multipliers
        self.seed = int(seed)
        self.epoch = 0

        min_unit = self.sources_per_class * self.samples_per_source
        if min_unit > self.batch_size:
            self.samples_per_source = max(1, self.batch_size // self.sources_per_class)
            min_unit = self.sources_per_class * self.samples_per_source
        if self.classes_per_batch * min_unit > self.batch_size:
            self.classes_per_batch = max(1, self.batch_size // min_unit)

        self.indices_by_label_source = defaultdict(lambda: defaultdict(list))
        for idx, record in enumerate(records):
            label = int(record["label"])
            source_id = int(record.get("source_id", -1))
            self.indices_by_label_source[label][source_id].append(idx)

        self.labels = sorted(self.indices_by_label_source)
        if not self.labels:
            raise ValueError("SourceAwareBatchSampler requires at least one training record.")
        self.label_weights = self._build_label_weights()
        self.num_batches = max(1, math.ceil(len(records) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def _build_label_weights(self):
        if self.class_multipliers is None:
            return {label: 1.0 for label in self.labels}
        if len(self.class_multipliers) <= max(self.labels):
            raise ValueError(
                f"class_multipliers must cover label {max(self.labels)}, "
                f"got {len(self.class_multipliers)} values"
            )
        weights = {}
        for label in self.labels:
            weights[label] = max(0.0, float(self.class_multipliers[label]))
        if sum(weights.values()) <= 0.0:
            raise ValueError("At least one source-aware class multiplier must be positive.")
        return weights

    def _weighted_label_choice(self, rng, labels):
        total_weight = sum(self.label_weights.get(label, 1.0) for label in labels)
        if total_weight <= 0.0:
            return rng.choice(labels)
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for label in labels:
            cumulative += self.label_weights.get(label, 1.0)
            if cumulative >= threshold:
                return label
        return labels[-1]

    def _sample_labels(self, rng):
        if len(self.labels) >= self.classes_per_batch:
            available = list(self.labels)
            labels = []
            for _ in range(self.classes_per_batch):
                label = self._weighted_label_choice(rng, available)
                labels.append(label)
                available.remove(label)
            return labels
        return [self._weighted_label_choice(rng, self.labels) for _ in range(self.classes_per_batch)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.num_batches):
            labels = self._sample_labels(rng)

            batch = []
            for label in labels:
                by_source = self.indices_by_label_source[label]
                source_ids = sorted(by_source)
                if len(source_ids) >= self.sources_per_class:
                    chosen_sources = rng.sample(source_ids, self.sources_per_class)
                else:
                    chosen_sources = [rng.choice(source_ids) for _ in range(self.sources_per_class)]

                for source_id in chosen_sources:
                    indices = by_source[source_id]
                    if len(indices) >= self.samples_per_source:
                        batch.extend(rng.sample(indices, self.samples_per_source))
                    else:
                        batch.extend(rng.choice(indices) for _ in range(self.samples_per_source))

            while len(batch) < self.batch_size:
                label = rng.choice(self.labels)
                source_id = rng.choice(sorted(self.indices_by_label_source[label]))
                batch.append(rng.choice(self.indices_by_label_source[label][source_id]))
            rng.shuffle(batch)
            yield batch[:self.batch_size]


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce.detach()).clamp(1e-6, 1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def main():
    parser = argparse.ArgumentParser(
        description="Train environmental sound CNN (DS-Conv2D-H1 / DS-Res1D-SE / TCAM-Attn1D)"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset folder; defaults by config dataset")
    parser.add_argument("--config", type=str, default="configs/rtx3090_config.json", help="Path to RTX 3090 config JSON")
    parser.add_argument("--fold", type=int, default=1, help="Test fold/bucket. US8K uses 1-10; ESC-50 uses 1-5; Speech Commands uses 1.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Physical batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name suffix (e.g. crossentropy, msle)")
    parser.add_argument("--max_train_clips", type=int, default=None, help="Limit train clips for smoke tests after splitting")
    parser.add_argument("--max_val_clips", type=int, default=None, help="Limit validation clips for smoke tests after splitting")
    parser.add_argument("--max_test_clips", type=int, default=None, help="Limit test clips for smoke tests after splitting")
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        choices=[
            "paper_9_1",
            "clean_8_1_1",
            "random_clip_9_1",
            "source_group_9_1",
            "source_group_8_1_1",
            "esc50_3_1_1_foldk_valnext_v1",
            "esc50_official_4_1_cv",
            "speech_commands_v2_official12",
        ],
        help="Evaluation protocol. US8K protocols remain unchanged; ESC-50 and Speech Commands protocols are Phase 1 dataset contracts."
    )
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Config file {args.config} not found! Using fallback defaults.")
        cfg = {
            "batch_size": 96,
            "accum_steps": 1,
            "epochs": 200,
            "lr": 2e-4,
            "num_workers": 6,
            "sample_rate": 16000,
            "frame_length": 8000,
            "cycles": 4,
            "seed": 83
        }

    # CLI Overrides
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.protocol is not None:
        cfg["protocol"] = args.protocol

    dataset_name = normalize_dataset_name(cfg.get("dataset", cfg.get("dataset_name", "urbansound8k")))
    data_dir = args.data_dir or cfg.get("data_dir") or default_data_dir(dataset_name)
    protocol = cfg.get("protocol", "paper_9_1").lower()
    supported_protocols = {
        "paper_9_1",
        "clean_8_1_1",
        "random_clip_9_1",
        "source_group_9_1",
        "source_group_8_1_1",
        "esc50_3_1_1_foldk_valnext_v1",
        "esc50_official_4_1_cv",
        "speech_commands_v2_official12",
    }
    if protocol not in supported_protocols:
        raise ValueError(
            f"Unsupported protocol '{protocol}'. Use 'paper_9_1', 'clean_8_1_1', "
            "'random_clip_9_1', 'source_group_9_1', 'source_group_8_1_1', "
            "'esc50_3_1_1_foldk_valnext_v1', 'esc50_official_4_1_cv', "
            "or 'speech_commands_v2_official12'."
        )
    validate_fold_for_protocol(protocol, args.fold)

    # Setup environment
    set_seed(cfg.get("seed", 83))
    prepare_dirs()
    
    if args.exp_name:
        exp_dir = f"experiments/{args.exp_name}/fold_{args.fold}"
        ckpt_dir = f"{exp_dir}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        best_ckpt_path = f"{ckpt_dir}/tcam_fold_{args.fold}_best.pt"
        history_path = f"{exp_dir}/history.json"
        metrics_path = f"{exp_dir}/metrics.json"
        predictions_path = f"{exp_dir}/predictions.json"
        
        def get_cycle_ckpt_path(cycle_id):
            return f"{ckpt_dir}/tcam_fold_{args.fold}_cycle_{cycle_id}.pt"
    else:
        best_ckpt_path = f"checkpoints/tcam_fold_{args.fold}_best.pt"
        history_path = f"logs/fold_{args.fold}_history.json"
        metrics_path = f"results/metrics/fold_{args.fold}_metrics.json"
        predictions_path = f"results/predictions/fold_{args.fold}_predictions.json"
        
        def get_cycle_ckpt_path(cycle_id):
            return f"checkpoints/tcam_fold_{args.fold}_cycle_{cycle_id}.pt"
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device designated for training: {device}")

    clip_seconds = float(cfg.get("clip_seconds", 4.0))
    clip_records, class_names, dataset_name = parse_audio_dataset(
        dataset_name,
        data_dir,
        sample_rate=cfg.get("sample_rate", 16000),
        clip_seconds=clip_seconds,
    )
    num_classes = len(class_names)
    if cfg.get("num_classes") is not None and int(cfg["num_classes"]) != num_classes:
        raise ValueError(
            f"Config num_classes={cfg['num_classes']} does not match dataset "
            f"{dataset_name} class count {num_classes}."
        )
    cfg["num_classes"] = num_classes

    # 2. Setup split.
    print(f"\n=================== TRAINING FOLD {args.fold} ({dataset_name} | {protocol}) ===================")
    train_clips, val_clips, test_records, val_fold, uses_validation, split_description = build_dataset_split(
        clip_records,
        dataset_name,
        protocol,
        args.fold,
        cfg.get("seed", 83),
    )
    print(f"Protocol: {split_description}")

    if args.max_train_clips is not None or args.max_val_clips is not None or args.max_test_clips is not None:
        original_counts = (len(train_clips), len(val_clips), len(test_records))
        train_clips = make_stratified_clip_subset(train_clips, args.max_train_clips, cfg.get("seed", 83) + 101)
        val_clips = make_stratified_clip_subset(val_clips, args.max_val_clips, cfg.get("seed", 83) + 202)
        test_records = make_stratified_clip_subset(test_records, args.max_test_clips, cfg.get("seed", 83) + 303)
        print(
            "Smoke subset active | "
            f"Train {original_counts[0]}->{len(train_clips)}, "
            f"Val {original_counts[1]}->{len(val_clips)}, "
            f"Test {original_counts[2]}->{len(test_records)}."
        )

    source_overlap = source_label_overlap_summary(train_clips, test_records)
    if source_overlap["count"] is not None:
        print(f"Source-label overlap (fsID+classID) between train/test: {source_overlap['count']}")
    if uses_validation:
        train_val_overlap = source_label_overlap_summary(train_clips, val_clips)
        val_test_overlap = source_label_overlap_summary(val_clips, test_records)
        print(f"Source-label overlap (fsID+classID) between train/val: {train_val_overlap['count']}")
        print(f"Source-label overlap (fsID+classID) between val/test: {val_test_overlap['count']}")
    
    random.shuffle(train_clips)

    frame_length = int(cfg.get("frame_length", 8000))
    frame_hop = int(cfg.get("frame_hop", frame_length // 2))
    frames_per_clip = cfg.get("frames_per_clip", None)
    if frames_per_clip is not None:
        frames_per_clip = int(frames_per_clip)
    clip_seconds = float(cfg.get("clip_seconds", 4.0))
    drop_silent_tail_frames = bool(cfg.get("drop_silent_tail_frames", False))
    eval_drop_silent_tail_frames = bool(cfg.get("eval_drop_silent_tail_frames", drop_silent_tail_frames))
    target_len = int(cfg.get("sample_rate", 16000) * clip_seconds)
    effective_frames_per_clip = frames_per_clip
    if effective_frames_per_clip is None:
        effective_frames_per_clip = max(1, math.floor((target_len - frame_length) / frame_hop) + 1)
    
    train_frames = generate_frame_records(
        train_clips,
        frame_length=frame_length,
        frame_hop=frame_hop,
        sample_rate=cfg.get("sample_rate", 16000),
        clip_seconds=clip_seconds,
        frames_per_clip=frames_per_clip,
        drop_silent_tail_frames=drop_silent_tail_frames,
    )
    val_frames = generate_frame_records(
        val_clips,
        frame_length=frame_length,
        frame_hop=frame_hop,
        sample_rate=cfg.get("sample_rate", 16000),
        clip_seconds=clip_seconds,
        frames_per_clip=frames_per_clip,
        drop_silent_tail_frames=drop_silent_tail_frames,
    ) if uses_validation else []
    
    print(f"Clips: Train={len(train_clips)}, Val={len(val_clips)}, Test={len(test_records)}")
    print(f"Frames: Train={len(train_frames)}, Val={len(val_frames)}")
    supervised_contrastive_cfg = cfg.get("supervised_contrastive", {})
    supervised_contrastive_enabled = bool(supervised_contrastive_cfg.get("enabled", False))
    standalone_source_batch_cfg = cfg.get("source_aware_batch_sampler", {})
    supcon_source_batch_cfg = supervised_contrastive_cfg.get("source_aware_batch_sampler", {})
    if standalone_source_batch_cfg.get("enabled", False):
        source_batch_cfg = standalone_source_batch_cfg
        source_batch_origin = "source_aware_batch_sampler"
    else:
        source_batch_cfg = supcon_source_batch_cfg
        source_batch_origin = "supervised_contrastive.source_aware_batch_sampler"
    use_source_batch_sampler = bool(source_batch_cfg.get("enabled", False))
    machinery_source_robust_cfg = cfg.get("machinery_source_robust", {}) or {}
    machinery_source_robust_enabled = bool(machinery_source_robust_cfg.get("enabled", False))
    needs_source_ids = (
        supervised_contrastive_enabled or use_source_batch_sampler or machinery_source_robust_enabled
    )
    if needs_source_ids:
        source_to_id = attach_source_ids(train_frames)
        print(
            "[Source Data Setup] "
            f"source_groups={len(source_to_id)} | "
            f"supcon={supervised_contrastive_enabled} | "
            f"source_batch_sampler={use_source_batch_sampler} | "
            f"machinery_source_robust={machinery_source_robust_enabled}"
        )

    # 3. Preload waveforms to RAM after optional smoke subsetting.
    selected_clip_records = train_clips + val_clips + test_records
    print(f"\nPre-loading and resampling {len({r['path'] for r in selected_clip_records})} audio clips to RAM...")
    cached_waveforms = {}
    start_preload = time.time()

    path_clip_seconds = {}
    for record in selected_clip_records:
        current = path_clip_seconds.get(record["path"], clip_seconds)
        path_clip_seconds[record["path"]] = None if record.get("cache_full_waveform") else current
    paths = sorted(path_clip_seconds)
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda p: load_audio_to_ram(p, cfg.get("sample_rate", 16000), path_clip_seconds[p]),
            paths,
        )
        for path, w in results:
            cached_waveforms[path] = w

    print(f"Pre-loading completed in {time.time() - start_preload:.2f} seconds! RAM Caching is active.")

    # Dataloader
    train_dataset = CachedUrbanSoundFrameDataset(
        train_frames,
        cached_waveforms,
        frame_length=frame_length,
        augment_cfg=cfg.get("augment", None),
        return_source_id=supervised_contrastive_enabled or machinery_source_robust_enabled,
    )
    
    num_workers = cfg.get("num_workers", 0)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    weighted_sampler_cfg = cfg.get("weighted_sampler", {})
    if use_source_batch_sampler:
        if weighted_sampler_cfg.get("enabled", False):
            raise ValueError("weighted_sampler and source_aware_batch_sampler cannot both be enabled.")
        batch_sampler = SourceAwareBatchSampler(
            train_frames,
            batch_size=cfg.get("batch_size", 96),
            classes_per_batch=source_batch_cfg.get("classes_per_batch", 8),
            sources_per_class=source_batch_cfg.get("sources_per_class", 2),
            samples_per_source=source_batch_cfg.get("samples_per_source", 4),
            class_multipliers=source_batch_cfg.get("class_multipliers"),
            seed=cfg.get("seed", 83),
        )
        loader_kwargs["batch_sampler"] = batch_sampler
        print(
            "[DataLoader Setup] SourceAwareBatchSampler enabled | "
            f"origin={source_batch_origin} | "
            f"batch_size={batch_sampler.batch_size} | "
            f"classes_per_batch={batch_sampler.classes_per_batch} | "
            f"sources_per_class={batch_sampler.sources_per_class} | "
            f"samples_per_source={batch_sampler.samples_per_source} | "
            f"class_multipliers={source_batch_cfg.get('class_multipliers')}"
        )
    else:
        loader_kwargs.update({
            "batch_size": cfg.get("batch_size", 96),
            "shuffle": True,
            "drop_last": True,
        })
        if len(train_dataset) < loader_kwargs["batch_size"]:
            loader_kwargs["drop_last"] = False
            print(
                "Train frame count is smaller than batch size; disabling drop_last "
                "to keep smoke-test DataLoader non-empty."
            )
        if weighted_sampler_cfg.get("enabled", False):
            sampler = make_weighted_sampler(
                train_frames,
                num_classes=num_classes,
                multipliers=weighted_sampler_cfg.get("class_multipliers"),
            )
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
            print(
                "[DataLoader Setup] WeightedRandomSampler enabled | "
                f"class_multipliers={weighted_sampler_cfg.get('class_multipliers')}"
            )
        
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Instantiate model, loss, optimizer, scaler
    model_name, model = build_model(cfg, num_classes=num_classes)
    model = model.to(device)
    initial_checkpoint_path = load_initial_model_weights(model, cfg, args.fold, device)
    input_transform = build_input_transform(cfg, device)
    model_input_channels = 1
    model_input_length = frame_length
    if input_transform is not None:
        input_transform.eval()
        with torch.no_grad():
            feature_sample = input_transform(torch.zeros(1, 1, frame_length, device=device))
        model_input_channels = int(feature_sample.shape[1])
        model_input_length = int(feature_sample.shape[-1])
    model_params = count_parameters(model)
    model_macs = estimate_conv_linear_macs(
        model,
        model_input_length,
        device,
        input_channels=model_input_channels,
    )
    model_macs_per_clip = model_macs * effective_frames_per_clip
    print(
        f"[Model Setup] model={model_name} | params={model_params['params_with_bias']:,} | "
        f"MACs/input={model_macs:,} | MACs/clip={model_macs_per_clip:,} | "
        f"frame_length={frame_length} | frames_per_clip={effective_frames_per_clip}"
    )
    enforce_deployment_budget(cfg, model_params, model_macs_per_clip)
    if input_transform is not None:
        print(
            f"[Feature Setup] input_features={cfg.get('input_features')} | "
            f"classifier_input_channels={model_input_channels} | classifier_input_length={model_input_length}"
        )
    distillation_cfg = cfg.get("distillation", {})
    teacher_name, teacher_model, teacher_checkpoint_path, teacher_config_path = load_teacher_model(
        cfg,
        distillation_cfg,
        args.fold,
        device,
        num_classes,
    )
    if teacher_model is not None:
        print(
            "[Distillation Setup] enabled=True | "
            f"teacher={teacher_name} | "
            f"checkpoint={teacher_checkpoint_path} | "
            f"weight={float(distillation_cfg.get('weight', 0.2)):g} | "
            f"temperature={float(distillation_cfg.get('temperature', 2.0)):g} | "
            f"protect_classes={distillation_cfg.get('protect_classes', [])} | "
            f"apply_to_mixup={bool(distillation_cfg.get('apply_to_mixup', False))}"
        )
    
    loss_type = cfg.get("loss", "crossentropy").lower()
    if loss_type == "msle":
        print("[Loss Setup] Using Mean Squared Logarithmic Error (MSLE) Loss.")
        class MSLELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
            def forward(self, logits, target):
                probs = F.softmax(logits, dim=-1)
                target_onehot = F.one_hot(target, num_classes=logits.size(-1)).float()
                return self.mse(torch.log1p(probs), torch.log1p(target_onehot))
        criterion = MSLELoss()
    else:
        if loss_type == "focal":
            print("[Loss Setup] Using Focal Cross Entropy Loss.")
        else:
            print("[Loss Setup] Using Cross Entropy Loss.")
        label_smoothing = float(cfg.get("label_smoothing", 0.0))
        class_weighting = cfg.get("class_weighting", "none").lower()
        ce_weights = None
        if class_weighting == "balanced":
            ce_weights = balanced_class_weights(train_clips, num_classes=num_classes, device=device)
            ce_weights = apply_class_multipliers(
                ce_weights,
                cfg.get("class_weight_multipliers"),
                device=device,
            )
            print(f"[Loss Setup] Balanced class weights: {[round(float(w), 4) for w in ce_weights.cpu()]}")
        elif class_weighting != "none":
            raise ValueError(f"Unsupported class_weighting '{class_weighting}'. Use 'none' or 'balanced'.")
        if loss_type == "focal":
            criterion = FocalLoss(
                weight=ce_weights,
                gamma=float(cfg.get("focal_gamma", 1.5)),
                label_smoothing=label_smoothing,
            )
            print(f"[Loss Setup] Focal gamma={float(cfg.get('focal_gamma', 1.5)):g}")
        elif loss_type == "crossentropy":
            criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unsupported loss '{loss_type}'. Use 'msle', 'crossentropy', or 'focal'.")

    use_amp = bool(cfg.get("amp", True))
    gradient_clip = cfg.get("gradient_clip", 5.0)
    if gradient_clip is not None:
        gradient_clip = float(gradient_clip)
    adam_eps = float(cfg.get("adam_eps", 1e-8))

    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer_name = cfg.get("optimizer", "adam").lower()
    momentum = float(cfg.get("momentum", 0.9))
    nesterov_enabled = optimizer_name == "nesterov" or (
        optimizer_name == "sgd" and bool(cfg.get("nesterov", False))
    )

    print(
        f"[Numeric Setup] AMP={use_amp} | Gradient Clip={gradient_clip} | "
        f"Optimizer={optimizer_name} | Weight Decay={weight_decay:g} | Adam eps={adam_eps:g}"
    )

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("lr", 2e-4),
            eps=adam_eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.get("lr", 2e-4),
            eps=adam_eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name in {"sgd", "nesterov"}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.get("lr", 2e-4),
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov_enabled,
        )
        print(f"[Optimizer Setup] momentum={momentum:g} | nesterov={nesterov_enabled}")
    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=cfg.get("lr", 1.0),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            "Use 'adam', 'adamw', 'sgd', 'nesterov', or 'adadelta'."
        )
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device.type else "cpu", enabled=use_amp)
    ema_cfg = cfg.get("ema", {})
    ema = ModelEMA(model, decay=ema_cfg.get("decay", 0.995)) if ema_cfg.get("enabled", False) else None
    use_ema_for_validation = bool(ema_cfg.get("use_for_validation", True)) and ema is not None
    if ema is not None:
        print(
            f"[EMA Setup] enabled=True | decay={float(ema_cfg.get('decay', 0.995)):g} | "
            f"use_for_validation={use_ema_for_validation}"
        )

    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion, scaler=scaler,
        device=device, accumulation_steps=cfg.get("accum_steps", 1),
        use_amp=use_amp,
        gradient_clip=gradient_clip,
        input_transform=input_transform,
        mixup_cfg=cfg.get("mixup", {}),
        hard_negative_margin_cfg=cfg.get("hard_negative_margin", {}),
        supervised_contrastive_cfg=supervised_contrastive_cfg,
        distillation_cfg=distillation_cfg,
        teacher_model=teacher_model,
        ema=ema,
        machinery_source_robust_cfg=machinery_source_robust_cfg,
    )
    if cfg.get("mixup", {}).get("enabled", False):
        print(
            f"[Mixup Setup] enabled=True | alpha={float(cfg.get('mixup', {}).get('alpha', 0.2)):g} | "
            f"prob={float(cfg.get('mixup', {}).get('prob', 1.0)):g}"
        )
    hard_negative_cfg = cfg.get("hard_negative_margin", {})
    if hard_negative_cfg.get("enabled", False):
        print(
            "[Hard Negative Setup] enabled=True | "
            f"weight={float(hard_negative_cfg.get('weight', 0.05)):g} | "
            f"margin={float(hard_negative_cfg.get('margin', 0.5)):g} | "
            f"apply_to_mixup={bool(hard_negative_cfg.get('apply_to_mixup', False))} | "
            f"groups={hard_negative_cfg.get('groups', [])} | "
            f"pairs={hard_negative_cfg.get('pairs', [])}"
        )
    if machinery_source_robust_cfg.get("enabled", False):
        print(
            "[MC-ISR Setup] machinery_source_robust enabled | "
            f"source_weight={float(machinery_source_robust_cfg.get('source_weight', 0.15)):g} | "
            f"source_temperature={float(machinery_source_robust_cfg.get('source_temperature', 0.1)):g} | "
            f"machinery_classes={machinery_source_robust_cfg.get('machinery_classes', [0, 4, 5, 7])} | "
            f"apply_to_mixup={bool(machinery_source_robust_cfg.get('apply_to_mixup', False))}"
        )
    if supervised_contrastive_cfg.get("enabled", False):
        print(
            "[SupCon Setup] enabled=True | "
            f"weight={float(supervised_contrastive_cfg.get('weight', 0.05)):g} | "
            f"temperature={float(supervised_contrastive_cfg.get('temperature', 0.1)):g} | "
            f"source_aware={bool(supervised_contrastive_cfg.get('source_aware', True))} | "
            f"apply_to_mixup={bool(supervised_contrastive_cfg.get('apply_to_mixup', False))}"
        )

    best_acc = None
    history = {"train_loss": [], "train_acc": [], "val_clip_acc": []}
    
    cycles = cfg.get("cycles", 4)
    epochs = cfg.get("epochs", 200)
    epochs_per_cycle = math.ceil(epochs / cycles)
    print(f"[LR Schedule Setup] {describe_lr_schedule(cfg, cycles)}")
    snapshot_checkpoints = []
    snapshot_epochs = []
    early_stopping_cfg = cfg.get("early_stopping", {}) or {}
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False)) and uses_validation
    early_stopping_patience = int(early_stopping_cfg.get("patience", 30))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_warmup_epochs = int(early_stopping_cfg.get("warmup_epochs", epochs_per_cycle))
    early_stopping_best = None
    early_stopping_bad_epochs = 0
    early_stopped = False
    early_stop_epoch = None
    completed_epochs = 0
    if early_stopping_cfg.get("enabled", False):
        if uses_validation:
            print(
                "[Early Stopping Setup] enabled=True | monitor=val_clip_acc | mode=max | "
                f"warmup_epochs={early_stopping_warmup_epochs} | "
                f"patience={early_stopping_patience} | min_delta={early_stopping_min_delta:g}"
            )
        else:
            print("[Early Stopping Setup] disabled because this protocol has no validation split.")

    # Training Loop
    for epoch in range(epochs):
        lr = get_epoch_lr(epoch, epochs, cfg.get("lr", 2e-4), cycles, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        epoch_start = time.time()
        loss, train_acc = trainer.train_epoch(train_loader)
        completed_epochs = epoch + 1
        
        eval_model = ema.module if use_ema_for_validation else model
        if uses_validation:
            val_clip_acc = trainer.evaluate_clips(
                [eval_model],
                val_clips,
                cached_waveforms,
                frame_length=frame_length,
                frame_hop=frame_hop,
                frames_per_clip=effective_frames_per_clip,
                drop_silent_tail_frames=eval_drop_silent_tail_frames,
                sample_rate=cfg.get("sample_rate", 16000),
            )
        else:
            val_clip_acc = None
        
        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["val_clip_acc"].append(val_clip_acc)
        
        if uses_validation:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | LR={lr:.6f} | Train Loss={loss:.4f} | Train Acc={train_acc*100:.2f}% | Val Clip Acc={val_clip_acc*100:.2f}% | Time={time.time() - epoch_start:.2f}s")
        else:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | LR={lr:.6f} | Train Loss={loss:.4f} | Train Acc={train_acc*100:.2f}% | Time={time.time() - epoch_start:.2f}s")
        
        # Save best validation checkpoint only for the clean validation protocol.
        if uses_validation and (best_acc is None or val_clip_acc > best_acc):
            best_acc = val_clip_acc
            torch.save({
                "model_state_dict": eval_model.state_dict(),
                "epoch": epoch,
                "val_acc": val_clip_acc,
                "ema": use_ema_for_validation,
            }, best_ckpt_path)
            
        # Snapshot Ensemble saving
        if (epoch + 1) % epochs_per_cycle == 0:
            cycle_id = (epoch + 1) // epochs_per_cycle
            snapshot_path = get_cycle_ckpt_path(cycle_id)
            torch.save(eval_model.state_dict(), snapshot_path)
            snapshot_checkpoints.append(snapshot_path)
            snapshot_epochs.append(epoch + 1)
            print(f"--> Saved Snapshot Cycle {cycle_id} checkpoint.")

        if early_stopping_enabled and completed_epochs >= early_stopping_warmup_epochs:
            if early_stopping_best is None or val_clip_acc > early_stopping_best + early_stopping_min_delta:
                early_stopping_best = val_clip_acc
                early_stopping_bad_epochs = 0
            else:
                early_stopping_bad_epochs += 1
                if early_stopping_bad_epochs >= early_stopping_patience:
                    early_stopped = True
                    early_stop_epoch = completed_epochs
                    print(
                        "--> Early stopping triggered: "
                        f"epoch={completed_epochs}, best_val={early_stopping_best*100:.2f}%, "
                        f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta:g}."
                    )
                    break

    final_epoch = completed_epochs or epochs
    if not snapshot_epochs or snapshot_epochs[-1] != final_epoch:
        eval_model = ema.module if use_ema_for_validation else model
        snapshot_path = get_cycle_ckpt_path("final")
        torch.save(eval_model.state_dict(), snapshot_path)
        snapshot_checkpoints.append(snapshot_path)
        snapshot_epochs.append(final_epoch)
        print(f"--> Saved Final Epoch checkpoint at epoch {final_epoch}.")

    # Load and evaluate the best validation model only when the protocol has a validation fold.
    if uses_validation and os.path.exists(best_ckpt_path):
        _, best_model = build_model(cfg, num_classes=num_classes)
        best_model = best_model.to(device)
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        best_model.load_state_dict(best_ckpt["model_state_dict"] if "model_state_dict" in best_ckpt else best_ckpt)
        test_acc_best, preds_best = trainer.evaluate_clips(
            [best_model],
            test_records,
            cached_waveforms,
            frame_length=frame_length,
            frame_hop=frame_hop,
            frames_per_clip=effective_frames_per_clip,
            drop_silent_tail_frames=eval_drop_silent_tail_frames,
            sample_rate=cfg.get("sample_rate", 16000),
            return_predictions=True,
        )
    else:
        test_acc_best, preds_best = None, []

    # Ensemble Evaluation
    if not snapshot_checkpoints:
        raise RuntimeError("No snapshot checkpoints were saved; cannot evaluate final snapshot or ensemble.")

    ensemble_models = []
    for i in range(len(snapshot_checkpoints) - 1, max(-1, len(snapshot_checkpoints) - 3), -1):
        _, m = build_model(cfg, num_classes=num_classes)
        m = m.to(device)
        m.load_state_dict(torch.load(snapshot_checkpoints[i], weights_only=True))
        ensemble_models.append(m)
        
    last_snapshot_epoch = snapshot_epochs[-1]
    test_acc_last, preds_last = trainer.evaluate_clips(
        [ensemble_models[0]],
        test_records,
        cached_waveforms,
        frame_length=frame_length,
        frame_hop=frame_hop,
        frames_per_clip=effective_frames_per_clip,
        drop_silent_tail_frames=eval_drop_silent_tail_frames,
        sample_rate=cfg.get("sample_rate", 16000),
        return_predictions=True,
    )
    test_acc_ensemble, preds_ensemble = trainer.evaluate_clips(
        ensemble_models,
        test_records,
        cached_waveforms,
        frame_length=frame_length,
        frame_hop=frame_hop,
        frames_per_clip=effective_frames_per_clip,
        drop_silent_tail_frames=eval_drop_silent_tail_frames,
        sample_rate=cfg.get("sample_rate", 16000),
        return_predictions=True,
    )
    
    print(f"\n=================== FOLD {args.fold} FINAL EVALUATION RESULTS ===================")
    if uses_validation:
        print(f"  Best Validation Model Test Accuracy: {test_acc_best*100:.2f}%")
    else:
        print("  Best Validation Model Test Accuracy: N/A (paper_9_1 uses no validation fold)")
    print(f"  Last Snapshot (Epoch {last_snapshot_epoch}) Test Accuracy: {test_acc_last*100:.2f}%")
    print(f"  Ensembled Model (Last 2 Cycles) Test Accuracy: {test_acc_ensemble*100:.2f}%")
    
    # Save training history logs
    with open(history_path, "w") as fh:
        json.dump(history, fh)

    # Get git commit hash
    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    # Save metrics JSON
    metrics = {
        "fold": args.fold,
        "protocol": protocol,
        "random_split_algorithm": RANDOM_SPLIT_ALGORITHM if protocol == "random_clip_9_1" else None,
        "source_group_split_algorithm": SOURCE_GROUP_SPLIT_ALGORITHM if protocol in {"source_group_9_1", "source_group_8_1_1"} else None,
        "uses_validation": uses_validation,
        "dataset": dataset_name,
        "data_dir": data_dir,
        "class_names": class_names,
        "num_classes": num_classes,
        "official_train_folds": sorted_unique(r["fold"] for r in train_clips),
        "val_fold": val_fold,
        "test_fold": args.fold,
        "official_test_folds": sorted_unique(r["fold"] for r in test_records),
        "source_label_overlap_train_test": source_overlap,
        "train_clip_count": len(train_clips),
        "val_clip_count": len(val_clips),
        "test_clip_count": len(test_records),
        "train_frame_count": len(train_frames),
        "val_frame_count": len(val_frames),
        "model_name": model_name,
        "model_params": model_params,
        "model_conv_linear_macs_per_input": model_macs,
        "model_conv_linear_macs_per_clip_eval": model_macs_per_clip,
        "deployment_budget": cfg.get("deployment_budget"),
        "input_features": cfg.get("input_features", "waveform"),
        "classifier_input_channels": model_input_channels,
        "classifier_input_length": model_input_length,
        "frame_length": frame_length,
        "frame_hop": frame_hop,
        "frames_per_clip": effective_frames_per_clip,
        "drop_silent_tail_frames": drop_silent_tail_frames,
        "eval_drop_silent_tail_frames": eval_drop_silent_tail_frames,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "cycles": cycles,
        "snapshot_epochs": snapshot_epochs,
        "last_snapshot_epoch": last_snapshot_epoch,
        "early_stopping": cfg.get("early_stopping"),
        "early_stopped": early_stopped,
        "early_stop_epoch": early_stop_epoch,
        "early_stopping_best_val_acc": early_stopping_best,
        "seed": cfg.get("seed", 83),
        "loss_type": loss_type,
        "focal_gamma": cfg.get("focal_gamma"),
        "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
        "class_weighting": cfg.get("class_weighting", "none"),
        "class_weight_multipliers": cfg.get("class_weight_multipliers"),
        "weighted_sampler": cfg.get("weighted_sampler"),
        "source_aware_batch_sampler": cfg.get("source_aware_batch_sampler"),
        "pool_type": cfg.get("pool_type", "avg"),
        "pool_bins": cfg.get("pool_bins"),
        "stem_type": cfg.get("stem_type", "single"),
        "extra_late_blocks": cfg.get("extra_late_blocks", 0),
        "mixup": cfg.get("mixup"),
        "supervised_contrastive": cfg.get("supervised_contrastive"),
        "hard_negative_margin": cfg.get("hard_negative_margin"),
        "distillation": cfg.get("distillation"),
        "initial_checkpoint": cfg.get("initial_checkpoint"),
        "initial_checkpoint_path": initial_checkpoint_path,
        "teacher_checkpoint_path": teacher_checkpoint_path,
        "teacher_config_path": teacher_config_path,
        "ema": cfg.get("ema"),
        "use_ema_for_validation": use_ema_for_validation,
        "amp": use_amp,
        "gradient_clip": gradient_clip,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "adam_eps": adam_eps,
        "momentum": momentum,
        "nesterov": nesterov_enabled,
        "lr_schedule": cfg.get("lr_schedule", {"type": "cosine_restart"}),
        "batch_size": cfg.get("batch_size", 96),
        "accum_steps": cfg.get("accum_steps", 1),
        "max_train_clips": args.max_train_clips,
        "max_val_clips": args.max_val_clips,
        "max_test_clips": args.max_test_clips,
        "config_path": args.config,
        "git_commit": git_commit,
        "best_val_clip_acc": best_acc,
        "test_acc_best_val_model": test_acc_best,
        "test_acc_last_snapshot": test_acc_last,
        "test_acc_ensemble": test_acc_ensemble
    }
    with open(metrics_path, "w") as fm:
        json.dump(metrics, fm, indent=2)

    # Save predictions JSON
    preds_data = {
        "best_val_model_predictions": preds_best,
        "last_snapshot_predictions": preds_last,
        "ensemble_model_predictions": preds_ensemble
    }
    with open(predictions_path, "w") as fp:
        json.dump(preds_data, fp, indent=2)

if __name__ == "__main__":
    main()
