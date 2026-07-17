import os
import csv
import math
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


def _db_to_amplitude(db):
    return 10.0 ** (db / 20.0)


class WaveformAugment:
    """
    Lightweight raw-waveform augmentation for proposed models. It intentionally
    avoids expensive transforms so it remains suitable for low-latency 1D-CNN
    experiments and server-side sweeps.
    """
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.gain_db = float(cfg.get("gain_db", 0.0))
        self.noise_prob = float(cfg.get("noise_prob", 0.0))
        self.noise_snr_db_min = float(cfg.get("noise_snr_db_min", 20.0))
        self.noise_snr_db_max = float(cfg.get("noise_snr_db_max", 35.0))
        self.shift_prob = float(cfg.get("shift_prob", 0.0))
        self.shift_max_fraction = float(cfg.get("shift_max_fraction", 0.0))
        self.polarity_prob = float(cfg.get("polarity_prob", 0.0))
        self.time_mask_prob = float(cfg.get("time_mask_prob", 0.0))
        self.time_mask_max_fraction = float(cfg.get("time_mask_max_fraction", 0.0))

    def __call__(self, x):
        if not self.enabled:
            return x

        if self.gain_db > 0.0:
            db = random.uniform(-self.gain_db, self.gain_db)
            x = x * _db_to_amplitude(db)

        if self.polarity_prob > 0.0 and random.random() < self.polarity_prob:
            x = -x

        if self.shift_prob > 0.0 and self.shift_max_fraction > 0.0 and random.random() < self.shift_prob:
            max_shift = int(x.shape[-1] * self.shift_max_fraction)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                if shift > 0:
                    x = F.pad(x[..., :-shift], (shift, 0))
                elif shift < 0:
                    x = F.pad(x[..., -shift:], (0, -shift))

        if self.time_mask_prob > 0.0 and self.time_mask_max_fraction > 0.0 and random.random() < self.time_mask_prob:
            max_width = int(x.shape[-1] * self.time_mask_max_fraction)
            if max_width > 0:
                width = random.randint(1, max_width)
                start = random.randint(0, max(0, x.shape[-1] - width))
                x = x.clone()
                x[..., start:start + width] = 0.0

        if self.noise_prob > 0.0 and random.random() < self.noise_prob:
            rms = torch.sqrt(torch.mean(x * x).clamp_min(1e-12))
            snr_db = random.uniform(self.noise_snr_db_min, self.noise_snr_db_max)
            noise_rms = rms / _db_to_amplitude(snr_db)
            x = x + torch.randn_like(x) * noise_rms

        return torch.clamp(x, -1.0, 1.0)

class CachedUrbanSoundFrameDataset(Dataset):
    """
    Retrieves waveforms directly from pre-loaded RAM dictionary,
    extracting 8000-sample frames on-the-fly. No Disk I/O during training.
    """
    def __init__(self, records, cached_waveforms, frame_length=8000, augment_cfg=None):
        self.records = records
        self.cached_waveforms = cached_waveforms
        self.frame_length = frame_length
        self.augment = WaveformAugment(augment_cfg)

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

        frame = self.augment(frame.float())
            
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
        start_idx = header.index("start") if "start" in header else None
        end_idx = header.index("end") if "end" in header else None
        
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
            if start_idx is not None and end_idx is not None:
                start_s = float(row[start_idx])
                end_s = float(row[end_idx])
                record["start"] = start_s
                record["end"] = end_s
                record["duration"] = max(0.0, end_s - start_s)
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

def generate_frame_records(
    clip_records,
    frame_length=8000,
    frame_hop=None,
    sample_rate=16000,
    clip_seconds=4.0,
    frames_per_clip=None,
    drop_silent_tail_frames=False,
):
    """
    Expands clip records into frame records. Defaults preserve the paper-style
    15 frames per 4s clip with 50% overlap for 8000-sample frames.
    """
    if frame_hop is None:
        frame_hop = frame_length // 2
    if frames_per_clip is None:
        target_len = int(sample_rate * clip_seconds)
        frames_per_clip = max(1, math.floor((target_len - frame_length) / frame_hop) + 1)

    frame_records = []
    for r in clip_records:
        duration_samples = None
        if drop_silent_tail_frames and "duration" in r:
            duration_samples = int(float(r["duration"]) * sample_rate)
        for i in range(frames_per_clip):
            frame_start = i * frame_hop
            if duration_samples is not None and frame_start >= duration_samples:
                continue
            frame_records.append({
                "path": r["path"],
                "label": r["label"],
                "fold": r["fold"],
                "frame_start": frame_start
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
