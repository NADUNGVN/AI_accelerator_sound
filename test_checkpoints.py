import os
import sys
import argparse
import torch

# Ensure local src directory is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import TCAM1DCNN
from src.data import parse_dataset, load_audio_to_ram
from src.training import Trainer

def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on Fold 1 Test Set")
    parser.add_argument("--fold", type=int, default=1, help="Fold index to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Standard 10 classes
    class_names = [
        'air_conditioner','car_horn','children_playing','dog_bark',
        'drilling','engine_idling','gun_shot','jackhammer','siren','street_music'
    ]
    
    data_dir = "data/raw/UrbanSound8K"
    csv_path = os.path.join(data_dir, "metadata/UrbanSound8K.csv")
    audio_base = os.path.join(data_dir, "audio")
    
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found! Make sure you are in the repository root directory.")
        return
        
    # Parse dataset
    clip_records = parse_dataset(csv_path, audio_base, class_names)
    test_records = [r for r in clip_records if r["fold"] == args.fold]
    print(f"Loaded {len(test_records)} test clips for Fold {args.fold}")
    
    # Preload test waveforms
    print("Pre-loading test waveforms to RAM...")
    cached_waveforms = {}
    from concurrent.futures import ThreadPoolExecutor
    paths = [r["path"] for r in test_records]
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: load_audio_to_ram(p, 16000), paths)
        for path, w in results:
            cached_waveforms[path] = w
            
    # Instantiate trainer helper
    trainer = Trainer(model=None, optimizer=None, criterion=None, scaler=None, device=device)
    
    checkpoints_to_test = [
        ("Best Validation Checkpoint", f"checkpoints/tcam_fold_{args.fold}_best.pt"),
        ("Cycle 1 Snapshot", f"checkpoints/tcam_fold_{args.fold}_cycle_1.pt"),
        ("Cycle 2 Snapshot", f"checkpoints/tcam_fold_{args.fold}_cycle_2.pt"),
        ("Cycle 3 Snapshot", f"checkpoints/tcam_fold_{args.fold}_cycle_3.pt"),
        ("Cycle 4 Snapshot", f"checkpoints/tcam_fold_{args.fold}_cycle_4.pt"),
    ]
    
    print("\n=================== EVALUATING SINGLE CHECKPOINTS ===================")
    models_dict = {}
    for name, path in checkpoints_to_test:
        if not os.path.exists(path):
            print(f"{name} ({path}) not found, skipping.")
            continue
        model = TCAM1DCNN(num_classes=10).to(device)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        
        # Handle different saving wrappers
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
            
        model.eval()
        models_dict[name] = model
        
        acc = trainer.evaluate_clips([model], test_records, cached_waveforms, frame_length=8000)
        print(f"  * {name:<30}: {acc*100:.2f}%")
        
    print("\n=================== EVALUATING SNAPSHOT ENSEMBLES ===================")
    # Ensemble Last 2 Cycles
    ensemble_last_2 = []
    for name in ["Cycle 3 Snapshot", "Cycle 4 Snapshot"]:
        if name in models_dict:
            ensemble_last_2.append(models_dict[name])
    if len(ensemble_last_2) == 2:
        acc = trainer.evaluate_clips(ensemble_last_2, test_records, cached_waveforms, frame_length=8000)
        print(f"  * Ensemble Last 2 Cycles (Cycle 3 + 4) : {acc*100:.2f}%")
        
    # Ensemble All 4 Cycles
    ensemble_all_4 = []
    for name in ["Cycle 1 Snapshot", "Cycle 2 Snapshot", "Cycle 3 Snapshot", "Cycle 4 Snapshot"]:
        if name in models_dict:
            ensemble_all_4.append(models_dict[name])
    if len(ensemble_all_4) > 0:
        acc = trainer.evaluate_clips(ensemble_all_4, test_records, cached_waveforms, frame_length=8000)
        print(f"  * Ensemble All Available Cycles ({len(ensemble_all_4)} models): {acc*100:.2f}%")

if __name__ == "__main__":
    main()
