import os
import sys
import argparse
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Ensure local src directory is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import TCAM1DCNN, EfficientAudioCNN1D, KV260AudioNetDS1D
from src.data import CachedUrbanSoundFrameDataset, parse_dataset, generate_frame_records, load_audio_to_ram
from src.training import Trainer
from src.utils import set_seed, prepare_dirs


RANDOM_SPLIT_ALGORITHM = "stable_metadata_v2"
SOURCE_GROUP_SPLIT_ALGORITHM = "fsid_classid_balanced_v1"


def default_data_dir():
    """
    Prefer the repo-local layout, but also support the shared research dataset
    layout used by this workspace.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repo_root, "data", "raw", "UrbanSound8K"),
        os.path.abspath(os.path.join(repo_root, "..", "..", "..", "data", "UrbanSound8K")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "metadata", "UrbanSound8K.csv")):
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

    rng = random.Random(seed)
    groups_by_class = defaultdict(dict)
    for record in clip_records:
        key = (str(record["fsID"]), int(record["classID"]))
        groups_by_class[record["label"]].setdefault(key, []).append(record)

    train_clips = []
    test_records = []
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

        for idx, records in enumerate(buckets, start=1):
            if idx == test_bucket:
                test_records.extend(records)
            else:
                train_clips.extend(records)

    return train_clips, test_records


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


def build_model(cfg, num_classes):
    model_name = cfg.get("model_name", "tcam1dcnn").lower()
    if model_name == "tcam1dcnn":
        model = TCAM1DCNN(num_classes=num_classes)
    elif model_name == "efficient_audio_cnn1d":
        model = EfficientAudioCNN1D(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.25)),
        )
    elif model_name == "kv260_audio_net_ds1d":
        model = KV260AudioNetDS1D(
            num_classes=num_classes,
            width_mult=float(cfg.get("width_mult", 1.0)),
            dropout=float(cfg.get("dropout", 0.15)),
        )
    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Use 'tcam1dcnn', "
            "'efficient_audio_cnn1d', or 'kv260_audio_net_ds1d'."
        )
    return model_name, model


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


def estimate_conv_linear_macs(model, input_length, device):
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
        model(torch.zeros(1, 1, input_length, device=device))
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return macs


def balanced_class_weights(records, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for record in records:
        counts[int(record["label"])] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train SOTA TCAM1DCNN on RTX 3090 (Modular Structure)")
    parser.add_argument("--data_dir", type=str, default=default_data_dir(), help="Path to UrbanSound8K folder")
    parser.add_argument("--config", type=str, default="configs/rtx3090_config.json", help="Path to RTX 3090 config JSON")
    parser.add_argument("--fold", type=int, default=1, help="Test fold for 10-fold CV (1-10)")
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
        choices=["paper_9_1", "clean_8_1_1", "random_clip_9_1", "source_group_9_1"],
        help="Evaluation protocol. paper_9_1 uses official folds; clean_8_1_1 keeps a validation fold; random_clip_9_1 is a stratified random clip-level control; source_group_9_1 is a source-label-grouped random control."
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

    protocol = cfg.get("protocol", "paper_9_1").lower()
    if protocol not in {"paper_9_1", "clean_8_1_1", "random_clip_9_1", "source_group_9_1"}:
        raise ValueError(
            f"Unsupported protocol '{protocol}'. Use 'paper_9_1', 'clean_8_1_1', "
            "'random_clip_9_1', or 'source_group_9_1'."
        )
    if not 1 <= args.fold <= 10:
        raise ValueError(f"--fold must be in [1, 10], got {args.fold}")

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

    # Standard 10 classes matching paper
    class_names = [
        'air_conditioner','car_horn','children_playing','dog_bark',
        'drilling','engine_idling','gun_shot','jackhammer','siren','street_music'
    ]

    csv_path = os.path.join(args.data_dir, "metadata/UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")

    # 1. Parse dataset (8732 standard clips, rail_vehicle filtered)
    clip_records = parse_dataset(csv_path, audio_base, class_names)

    # 2. Setup split.
    print(f"\n=================== TRAINING FOLD {args.fold} ({protocol}) ===================")
    val_fold = None
    val_clips = []
    uses_validation = False

    if protocol == "paper_9_1":
        test_records = [r for r in clip_records if r["fold"] == args.fold]
        train_clips = [r for r in clip_records if r["fold"] != args.fold]
        print("Protocol: paper_9_1 | Train=9 folds, Test=1 fold, no validation-based model selection.")
    elif protocol == "clean_8_1_1":
        test_records = [r for r in clip_records if r["fold"] == args.fold]
        val_fold = (args.fold % 10) + 1
        train_clips = [r for r in clip_records if r["fold"] != args.fold and r["fold"] != val_fold]
        val_clips = [r for r in clip_records if r["fold"] == val_fold]
        uses_validation = True
        print(f"Protocol: clean_8_1_1 | Train=8 folds, Val=fold {val_fold}, Test=fold {args.fold}.")
    elif protocol == "random_clip_9_1":
        train_clips, test_records = make_stratified_random_clip_split(
            clip_records,
            test_bucket=args.fold,
            seed=cfg.get("seed", 83),
        )
        print(
            "Protocol: random_clip_9_1 | Stratified random clip-level 9/1 control, "
            f"Test bucket={args.fold}, seed={cfg.get('seed', 83)}, split_algorithm={RANDOM_SPLIT_ALGORITHM}."
        )
    else:
        train_clips, test_records = make_stratified_source_group_split(
            clip_records,
            test_bucket=args.fold,
            seed=cfg.get("seed", 83),
        )
        print(
            "Protocol: source_group_9_1 | Stratified random source-label-group 9/1 control, "
            f"Test bucket={args.fold}, seed={cfg.get('seed', 83)}, split_algorithm={SOURCE_GROUP_SPLIT_ALGORITHM}."
        )

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
    
    random.shuffle(train_clips)

    frame_length = int(cfg.get("frame_length", 8000))
    frame_hop = int(cfg.get("frame_hop", frame_length // 2))
    frames_per_clip = cfg.get("frames_per_clip", None)
    if frames_per_clip is not None:
        frames_per_clip = int(frames_per_clip)
    clip_seconds = float(cfg.get("clip_seconds", 4.0))
    drop_silent_tail_frames = bool(cfg.get("drop_silent_tail_frames", False))
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

    # 3. Preload waveforms to RAM after optional smoke subsetting.
    selected_clip_records = train_clips + val_clips + test_records
    print(f"\nPre-loading and resampling {len({r['path'] for r in selected_clip_records})} audio clips to RAM...")
    cached_waveforms = {}
    start_preload = time.time()

    paths = sorted({r["path"] for r in selected_clip_records})
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: load_audio_to_ram(p, cfg.get("sample_rate", 16000)), paths)
        for path, w in results:
            cached_waveforms[path] = w

    print(f"Pre-loading completed in {time.time() - start_preload:.2f} seconds! RAM Caching is active.")

    # Dataloader
    train_dataset = CachedUrbanSoundFrameDataset(
        train_frames,
        cached_waveforms,
        frame_length=frame_length,
        augment_cfg=cfg.get("augment", None),
    )
    
    num_workers = cfg.get("num_workers", 0)
    loader_kwargs = {
        "batch_size": cfg.get("batch_size", 96),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if len(train_dataset) < loader_kwargs["batch_size"]:
        loader_kwargs["drop_last"] = False
        print(
            "Train frame count is smaller than batch size; disabling drop_last "
            "to keep smoke-test DataLoader non-empty."
        )
        
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Instantiate model, loss, optimizer, scaler
    model_name, model = build_model(cfg, num_classes=10)
    model = model.to(device)
    model_params = count_parameters(model)
    model_macs = estimate_conv_linear_macs(model, frame_length, device)
    print(
        f"[Model Setup] model={model_name} | params={model_params['params_with_bias']:,} | "
        f"MACs/input={model_macs:,} | frame_length={frame_length} | frames_per_clip={effective_frames_per_clip}"
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
        print("[Loss Setup] Using Cross Entropy Loss.")
        label_smoothing = float(cfg.get("label_smoothing", 0.0))
        class_weighting = cfg.get("class_weighting", "none").lower()
        ce_weights = None
        if class_weighting == "balanced":
            ce_weights = balanced_class_weights(train_clips, num_classes=10, device=device)
            print(f"[Loss Setup] Balanced class weights: {[round(float(w), 4) for w in ce_weights.cpu()]}")
        elif class_weighting != "none":
            raise ValueError(f"Unsupported class_weighting '{class_weighting}'. Use 'none' or 'balanced'.")
        criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=label_smoothing)

    use_amp = bool(cfg.get("amp", True))
    gradient_clip = cfg.get("gradient_clip", 5.0)
    if gradient_clip is not None:
        gradient_clip = float(gradient_clip)
    adam_eps = float(cfg.get("adam_eps", 1e-8))

    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer_name = cfg.get("optimizer", "adam").lower()

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
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adam' or 'adamw'.")
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device.type else "cpu", enabled=use_amp)

    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion, scaler=scaler,
        device=device, accumulation_steps=cfg.get("accum_steps", 1),
        use_amp=use_amp, gradient_clip=gradient_clip
    )

    best_acc = None
    history = {"train_loss": [], "train_acc": [], "val_clip_acc": []}
    
    cycles = cfg.get("cycles", 4)
    epochs = cfg.get("epochs", 200)
    epochs_per_cycle = math.ceil(epochs / cycles)
    snapshot_checkpoints = []

    # Training Loop
    for epoch in range(epochs):
        lr = trainer.get_cosine_lr(epoch, epochs, cfg.get("lr", 2e-4), cycles)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        epoch_start = time.time()
        loss, train_acc = trainer.train_epoch(train_loader)
        
        if uses_validation:
            val_clip_acc = trainer.evaluate_clips(
                [model],
                val_clips,
                cached_waveforms,
                frame_length=frame_length,
                frame_hop=frame_hop,
                frames_per_clip=effective_frames_per_clip,
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
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_clip_acc
            }, best_ckpt_path)
            
        # Snapshot Ensemble saving
        if (epoch + 1) % epochs_per_cycle == 0:
            cycle_id = (epoch + 1) // epochs_per_cycle
            snapshot_path = get_cycle_ckpt_path(cycle_id)
            torch.save(model.state_dict(), snapshot_path)
            snapshot_checkpoints.append(snapshot_path)
            print(f"--> Saved Snapshot Cycle {cycle_id} checkpoint.")

    # Load and evaluate the best validation model only when the protocol has a validation fold.
    if uses_validation and os.path.exists(best_ckpt_path):
        _, best_model = build_model(cfg, num_classes=10)
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
            return_predictions=True,
        )
    else:
        test_acc_best, preds_best = None, []

    # Ensemble Evaluation
    if not snapshot_checkpoints:
        raise RuntimeError("No snapshot checkpoints were saved; cannot evaluate final snapshot or ensemble.")

    ensemble_models = []
    for i in range(len(snapshot_checkpoints) - 1, max(-1, len(snapshot_checkpoints) - 3), -1):
        _, m = build_model(cfg, num_classes=10)
        m = m.to(device)
        m.load_state_dict(torch.load(snapshot_checkpoints[i], weights_only=True))
        ensemble_models.append(m)
        
    test_acc_last, preds_last = trainer.evaluate_clips(
        [ensemble_models[0]],
        test_records,
        cached_waveforms,
        frame_length=frame_length,
        frame_hop=frame_hop,
        frames_per_clip=effective_frames_per_clip,
        return_predictions=True,
    )
    test_acc_ensemble, preds_ensemble = trainer.evaluate_clips(
        ensemble_models,
        test_records,
        cached_waveforms,
        frame_length=frame_length,
        frame_hop=frame_hop,
        frames_per_clip=effective_frames_per_clip,
        return_predictions=True,
    )
    
    print(f"\n=================== FOLD {args.fold} FINAL EVALUATION RESULTS ===================")
    if uses_validation:
        print(f"  Best Validation Model Test Accuracy: {test_acc_best*100:.2f}%")
    else:
        print("  Best Validation Model Test Accuracy: N/A (paper_9_1 uses no validation fold)")
    print(f"  Last Snapshot (Epoch {epochs}) Test Accuracy: {test_acc_last*100:.2f}%")
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
        "source_group_split_algorithm": SOURCE_GROUP_SPLIT_ALGORITHM if protocol == "source_group_9_1" else None,
        "uses_validation": uses_validation,
        "official_train_folds": sorted({r["fold"] for r in train_clips}),
        "val_fold": val_fold,
        "test_fold": args.fold,
        "official_test_folds": sorted({r["fold"] for r in test_records}),
        "source_label_overlap_train_test": source_overlap,
        "train_clip_count": len(train_clips),
        "val_clip_count": len(val_clips),
        "test_clip_count": len(test_records),
        "train_frame_count": len(train_frames),
        "val_frame_count": len(val_frames),
        "model_name": model_name,
        "model_params": model_params,
        "model_conv_linear_macs_per_input": model_macs,
        "model_conv_linear_macs_per_clip_eval": model_macs * effective_frames_per_clip,
        "frame_length": frame_length,
        "frame_hop": frame_hop,
        "frames_per_clip": effective_frames_per_clip,
        "drop_silent_tail_frames": drop_silent_tail_frames,
        "epochs": epochs,
        "cycles": cycles,
        "seed": cfg.get("seed", 83),
        "loss_type": loss_type,
        "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
        "class_weighting": cfg.get("class_weighting", "none"),
        "amp": use_amp,
        "gradient_clip": gradient_clip,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "adam_eps": adam_eps,
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
