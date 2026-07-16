import os
import sys
import argparse
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

# Ensure local src directory is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import TCAM1DCNN
from src.data import CachedUrbanSoundFrameDataset, parse_dataset, generate_frame_records, load_audio_to_ram
from src.training import Trainer
from src.utils import set_seed, prepare_dirs

def main():
    parser = argparse.ArgumentParser(description="Train SOTA TCAM1DCNN on RTX 3090 (Modular Structure)")
    parser.add_argument("--data_dir", type=str, default="data/raw/UrbanSound8K", help="Path to UrbanSound8K folder")
    parser.add_argument("--config", type=str, default="configs/rtx3090_config.json", help="Path to RTX 3090 config JSON")
    parser.add_argument("--fold", type=int, default=1, help="Test fold for 10-fold CV (1-10)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Physical batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name suffix (e.g. crossentropy, msle)")
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

    # 2. Preload waveforms to RAM (eliminating Disk I/O bottlenecks)
    print("\nPre-loading and resampling all audio clips to RAM (~2.2 GB memory)...")
    cached_waveforms = {}
    start_preload = time.time()
    
    paths = [r["path"] for r in clip_records]
    # Limit worker count to protect CPU overhead
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: load_audio_to_ram(p, cfg.get("sample_rate", 16000)), paths)
        for path, w in results:
            cached_waveforms[path] = w
            
    print(f"Pre-loading completed in {time.time() - start_preload:.2f} seconds! RAM Caching is active.")

    # 3. Setup Fold split (using strict predefined fold partitions to prevent source-level data leakage)
    print(f"\n=================== TRAINING OFFICIAL FOLD {args.fold} (RTX 3090 Configuration) ===================")
    val_fold = (args.fold % 10) + 1
    
    train_clips = [r for r in clip_records if r["fold"] != args.fold and r["fold"] != val_fold]
    val_clips = [r for r in clip_records if r["fold"] == val_fold]
    test_records = [r for r in clip_records if r["fold"] == args.fold]
    
    import random
    random.shuffle(train_clips)
    
    train_frames = generate_frame_records(train_clips)
    val_frames = generate_frame_records(val_clips)
    
    print(f"Clips: Train={len(train_clips)}, Val={len(val_clips)}, Test={len(test_records)}")
    print(f"Frames: Train={len(train_frames)}, Val={len(val_frames)}")

    # Dataloader
    train_dataset = CachedUrbanSoundFrameDataset(train_frames, cached_waveforms, frame_length=cfg.get("frame_length", 8000))
    
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
        
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Instantiate model, loss, optimizer, scaler
    model = TCAM1DCNN(num_classes=10).to(device)
    
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
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device.type else "cpu")

    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion, scaler=scaler,
        device=device, accumulation_steps=cfg.get("accum_steps", 1)
    )

    best_acc = 0.0
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
        
        # Periodic validation at the end of each epoch
        val_clip_acc = trainer.evaluate_clips([model], val_clips, cached_waveforms, frame_length=cfg.get("frame_length", 8000))
        
        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["val_clip_acc"].append(val_clip_acc)
        
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | LR={lr:.6f} | Train Loss={loss:.4f} | Train Acc={train_acc*100:.2f}% | Val Clip Acc={val_clip_acc*100:.2f}% | Time={time.time() - epoch_start:.2f}s")
        
        # Save best model checkpoint
        if val_clip_acc > best_acc:
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

    # Load and Evaluate Best Validation Model
    best_model = TCAM1DCNN(num_classes=10).to(device)
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        best_model.load_state_dict(best_ckpt["model_state_dict"] if "model_state_dict" in best_ckpt else best_ckpt)
        test_acc_best, preds_best = trainer.evaluate_clips([best_model], test_records, cached_waveforms, frame_length=cfg.get("frame_length", 8000), return_predictions=True)
    else:
        test_acc_best, preds_best = 0.0, []

    # Ensemble Evaluation
    ensemble_models = []
    for i in range(len(snapshot_checkpoints) - 1, max(-1, len(snapshot_checkpoints) - 3), -1):
        m = TCAM1DCNN(num_classes=10).to(device)
        m.load_state_dict(torch.load(snapshot_checkpoints[i], weights_only=True))
        ensemble_models.append(m)
        
    test_acc_last, preds_last = trainer.evaluate_clips([ensemble_models[0]], test_records, cached_waveforms, frame_length=cfg.get("frame_length", 8000), return_predictions=True)
    test_acc_ensemble, preds_ensemble = trainer.evaluate_clips(ensemble_models, test_records, cached_waveforms, frame_length=cfg.get("frame_length", 8000), return_predictions=True)
    
    print(f"\n=================== FOLD {args.fold} FINAL EVALUATION RESULTS ===================")
    print(f"  Best Validation Model Test Accuracy: {test_acc_best*100:.2f}%")
    print(f"  Last Snapshot (Epoch 200) Test Accuracy: {test_acc_last*100:.2f}%")
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
        "val_fold": val_fold,
        "epochs": epochs,
        "cycles": cycles,
        "seed": cfg.get("seed", 83),
        "loss_type": loss_type,
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
