import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Ensure local directories are on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import TCAM1DCNN
from src.data import parse_dataset, generate_frame_records, load_audio_to_ram, CachedUrbanSoundFrameDataset
from src.training import Trainer
from src.utils import set_seed, prepare_dirs

def main():
    print("=================== PIPELINE SANITY & INTEGRITY CHECK ===================")
    
    # 1. Configuration Setup
    set_seed(83)
    prepare_dirs()
    
    data_dir = "data/raw/UrbanSound8K"
    csv_path = os.path.join(data_dir, "metadata/UrbanSound8K.csv")
    audio_base = os.path.join(data_dir, "audio")
    
    class_names = [
        'air_conditioner','car_horn','children_playing','dog_bark',
        'drilling','engine_idling','gun_shot','jackhammer','siren','street_music'
    ]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV metadata file '{csv_path}' not found! Make sure you are in the correct repository directory.")
        sys.exit(1)
        
    # 2. Parse and Analyze Metadata
    print("\n--- [Step 1] Metadata Parsing & Class Distribution ---")
    clip_records = parse_dataset(csv_path, audio_base, class_names)
    
    class_counts = {name: 0 for name in class_names}
    for r in clip_records:
        class_counts[r["cls_name"]] += 1
        
    print("Class distribution in parsed clips:")
    for name, count in class_counts.items():
        print(f"  {name:20s}: {count} clips")
        
    # 3. Audio Reading & Integrity Scan (Checking for Silent/Zero Files)
    print("\n--- [Step 2] Audio Integrity Scan (Checking for Silent/Zero Files) ---")
    cached_waveforms = {}
    paths = [r["path"] for r in clip_records]
    
    print("Pre-loading and auditing audio files...")
    start_time = time.time()
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: load_audio_to_ram(p, 16000), paths)
        for path, w in results:
            cached_waveforms[path] = w
            
    print(f"Loaded {len(cached_waveforms)} clips in {time.time() - start_time:.2f}s.")
    
    silent_clips_count = 0
    extremely_quiet_count = 0
    non_silent_checked = 0
    
    for path, w in cached_waveforms.items():
        max_amplitude = np.max(np.abs(w))
        std_dev = np.std(w)
        
        if max_amplitude == 0:
            silent_clips_count += 1
        elif std_dev < 1e-4:
            extremely_quiet_count += 1
        else:
            non_silent_checked += 1
            
    print(f"\nAudit results:")
    print(f"  - Active/Normal clips : {non_silent_checked}")
    print(f"  - Completely silent (all zeros) clips : {silent_clips_count}")
    print(f"  - Extremely quiet clips (std < 1e-4)   : {extremely_quiet_count}")
    
    if silent_clips_count > 0:
        print("\n[WARNING] Critical Issue: Some audio files failed to load and were replaced with absolute silence (zeros).")
        print("This typically happens due to backend issues with torchaudio (e.g. missing soundfile/libsndfile package on the OS).")
    else:
        print("\n[SUCCESS] Audio integrity check passed! No silent dummy clips found.")
        
    # 4. Overfitting Sanity Test (Verify Learning Capacity)
    print("\n--- [Step 3] Overfitting Sanity Test on 50 Clips ---")
    # Take first 50 clips of train set
    test_fold = 1
    val_fold = 2
    train_clips_pool = [r for r in clip_records if r["fold"] != test_fold and r["fold"] != val_fold]
    subset_clips = train_clips_pool[:50]
    
    subset_frames = generate_frame_records(subset_clips)
    print(f"Selected subset: {len(subset_clips)} clips, expanded into {len(subset_frames)} frames.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for sanity training: {device}")
    
    model = TCAM1DCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Larger learning rate for fast overfitting
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device.type else "cpu")
    
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion, scaler=scaler,
        device=device, accumulation_steps=1
    )
    
    subset_dataset = CachedUrbanSoundFrameDataset(subset_frames, cached_waveforms, frame_length=8000)
    loader = torch.utils.data.DataLoader(
        subset_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    
    print("Training for 40 epochs to test if model can overfit the 50-clip subset...")
    overfitted_successfully = False
    for epoch in range(1, 41):
        loss, acc = trainer.train_epoch(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/40 | Loss = {loss:.4f} | Train Acc = {acc*100:.2f}%")
        if acc >= 0.98:
            print(f"\n[SUCCESS] Model successfully overfitted the training subset at epoch {epoch} (Acc = {acc*100:.2f}%)!")
            overfitted_successfully = True
            break
            
    if not overfitted_successfully:
        print("\n[FAIL] Model failed to overfit the small subset (Accuracy < 98% in 40 epochs).")
        print("This indicates a pipeline bug, an initialization issue, or vanishing gradients in the network architecture.")
        
    print("\n=================== SANITY CHECK COMPLETE ===================")

if __name__ == "__main__":
    main()
