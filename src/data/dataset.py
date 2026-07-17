import os
import csv
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class CachedUrbanSoundFrameDataset(Dataset):
    """
    Retrieves waveforms directly from pre-loaded RAM dictionary,
    extracting 8000-sample frames on-the-fly. No Disk I/O during training.
    """
    def __init__(self, records, cached_waveforms, frame_length=8000):
        self.records = records
        self.cached_waveforms = cached_waveforms
        self.frame_length = frame_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = record["path"]
        label = record["label"]
        frame_start = record["frame_start"]

        waveform_np = self.cached_waveforms[path]
        
        # Extract frame
        frame_np = waveform_np[:, frame_start:frame_start + self.frame_length]
        
        # Convert to tensor
        frame = torch.from_numpy(frame_np)
        
        # Verify length (failsafe)
        if frame.shape[-1] < self.frame_length:
            padding = self.frame_length - frame.shape[-1]
            frame = F.pad(frame, (0, padding), mode='constant')
            
        return frame, label

def parse_dataset(csv_path, audio_base_dir, class_names):
    """
    Parses UrbanSound8K CSV and returns standard 10-class records.
    Filters out the custom 11th class (rail_vehicle) to strictly replicate the paper.
    Fails fast on missing files or unexpected metadata because silently changing
    the dataset invalidates any comparison with the paper.
    """
    records = []
    class_map = {name: i for i, name in enumerate(class_names)}
    missing_paths = []
    unknown_classes = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        
        file_idx = header.index("slice_file_name")
        fold_idx = header.index("fold")
        class_idx = header.index("class")
        fsid_idx = header.index("fsID") if "fsID" in header else None
        class_id_idx = header.index("classID") if "classID" in header else None
        
        for row in r:
            filename = row[file_idx]
            fold = int(row[fold_idx])
            cls_name = row[class_idx]
            
            # Filter out rail_vehicle to strictly match paper (10 classes)
            if cls_name == "rail_vehicle":
                continue

            if cls_name not in class_map:
                unknown_classes.append(cls_name)
                continue
                
            path = os.path.join(audio_base_dir, f"fold{fold}", filename)
            exists = os.path.exists(path)
            if not exists:
                missing_paths.append(path)
                continue
                
            record = {
                "path": path,
                "slice_file_name": filename,
                "label": class_map[cls_name],
                "fold": fold,
                "cls_name": cls_name
            }
            if fsid_idx is not None:
                record["fsID"] = row[fsid_idx]
            if class_id_idx is not None:
                record["classID"] = int(row[class_id_idx])
            records.append(record)

    if unknown_classes:
        unique_unknown = sorted(set(unknown_classes))
        raise RuntimeError(
            f"Unknown classes in metadata: {unique_unknown}. "
            f"Expected one of {list(class_map)} plus optional rail_vehicle."
        )

    if missing_paths:
        examples = "\n".join(missing_paths[:20])
        raise RuntimeError(
            f"Metadata references {len(missing_paths)} missing audio files. "
            f"First missing paths:\n{examples}"
        )

    expected_count = 8732 if class_names == [
        'air_conditioner','car_horn','children_playing','dog_bark',
        'drilling','engine_idling','gun_shot','jackhammer','siren','street_music'
    ] else None
    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} UrbanSound8K 10-class clips after filtering, "
            f"but parsed {len(records)} clips from {csv_path}."
        )
            
    print(f"Parsed {len(records)} standard audio clips (10 classes, rail_vehicle excluded).")
    return records

def generate_frame_records(clip_records):
    """
    Expands clip records into frame records (15 frames per 4s clip with 50% overlap).
    """
    frame_records = []
    for r in clip_records:
        for i in range(15):
            frame_records.append({
                "path": r["path"],
                "label": r["label"],
                "fold": r["fold"],
                "frame_start": i * 4000
            })
    return frame_records

def load_audio_to_ram(path, sample_rate=16000):
    """
    Helper function to load, resample, and pad a single audio file to RAM.
    Raises on any decode or waveform-integrity problem. Replacing failed clips
    with silence creates mislabeled training examples and hides data issues.
    """
    try:
        waveform, source_sr = torchaudio.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode audio file: {path}") from exc

    if waveform.numel() == 0:
        raise RuntimeError(f"Decoded empty waveform: {path}")
    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"Decoded waveform contains NaN or Inf: {path}")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if source_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(source_sr, sample_rate)
        waveform = resampler(waveform)

    target_len = sample_rate * 4
    if waveform.shape[-1] < target_len:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]), mode='constant')
    else:
        waveform = waveform[:, :target_len]

    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"Preprocessed waveform contains NaN or Inf: {path}")
    if float(waveform.abs().max().item()) == 0.0:
        raise RuntimeError(f"Preprocessed waveform is completely silent: {path}")

    return path, waveform.contiguous().numpy()  # Convert to numpy array to prevent PyTorch shared memory leaks
