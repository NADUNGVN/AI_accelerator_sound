import os
import csv
import math
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


URBANSOUND8K_CLASS_NAMES = [
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

SPEECH_COMMANDS_WORDS = [
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
]
SPEECH_COMMANDS_SILENCE = "_silence_"
SPEECH_COMMANDS_UNKNOWN = "_unknown_"
SPEECH_COMMANDS_BACKGROUND_NOISE = "_background_noise_"
SPEECH_COMMANDS_CLASS_NAMES = SPEECH_COMMANDS_WORDS + [
    SPEECH_COMMANDS_SILENCE,
    SPEECH_COMMANDS_UNKNOWN,
]


def _db_to_amplitude(db):
    return 10.0 ** (db / 20.0)


def normalize_dataset_name(name):
    value = str(name or "urbansound8k").strip().lower().replace("-", "_")
    aliases = {
        "us8k": "urbansound8k",
        "urban_sound_8k": "urbansound8k",
        "urban_sound8k": "urbansound8k",
        "esc_50": "esc50",
        "speechcommands": "speech_commands",
        "speech_commands_v2": "speech_commands",
        "gsc": "speech_commands",
        "gsc_v2": "speech_commands",
    }
    return aliases.get(value, value)


def get_default_class_names(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "urbansound8k":
        return list(URBANSOUND8K_CLASS_NAMES)
    if dataset_name == "speech_commands":
        return list(SPEECH_COMMANDS_CLASS_NAMES)
    if dataset_name == "esc50":
        return None
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def _required_csv_field(header, field, csv_path):
    if field not in header:
        raise RuntimeError(f"Metadata file {csv_path} is missing required column '{field}'.")
    return header.index(field)


def _read_path_list(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Required split file not found: {path}")
    values = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip().replace("\\", "/")
            if item:
                values.add(item)
    return values


def _speech_speaker_id(filename):
    base = os.path.basename(filename)
    marker = "_nohash_"
    if marker in base:
        return base.split(marker, 1)[0]
    return os.path.splitext(base)[0]


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
    def __init__(
        self,
        records,
        cached_waveforms,
        frame_length=8000,
        augment_cfg=None,
        return_source_id=False,
    ):
        self.records = records
        self.cached_waveforms = cached_waveforms
        self.frame_length = frame_length
        self.augment = WaveformAugment(augment_cfg)
        self.return_source_id = bool(return_source_id)

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

        if self.return_source_id:
            return frame, label, int(record.get("source_id", -1))
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


def parse_esc50_dataset(data_dir):
    csv_path = os.path.join(data_dir, "meta", "esc50.csv")
    audio_base_dir = os.path.join(data_dir, "audio")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"ESC-50 metadata not found: {csv_path}")
    if not os.path.isdir(audio_base_dir):
        raise RuntimeError(f"ESC-50 audio directory not found: {audio_base_dir}")

    records = []
    target_to_category = {}
    missing_paths = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        filename_idx = _required_csv_field(header, "filename", csv_path)
        fold_idx = _required_csv_field(header, "fold", csv_path)
        target_idx = _required_csv_field(header, "target", csv_path)
        category_idx = _required_csv_field(header, "category", csv_path)
        src_file_idx = _required_csv_field(header, "src_file", csv_path)
        take_idx = header.index("take") if "take" in header else None

        for row in r:
            filename = row[filename_idx]
            fold = int(row[fold_idx])
            target = int(row[target_idx])
            category = row[category_idx]
            src_file = row[src_file_idx]
            target_to_category[target] = category
            path = os.path.join(audio_base_dir, filename)
            if not os.path.exists(path):
                missing_paths.append(path)
                continue
            record = {
                "path": path,
                "slice_file_name": filename,
                "label": target,
                "fold": fold,
                "cls_name": category,
                "fsID": src_file,
                "classID": target,
                "src_file": src_file,
                "clip_id": filename,
            }
            if take_idx is not None:
                record["take"] = row[take_idx]
            records.append(record)

    if missing_paths:
        examples = "\n".join(missing_paths[:20])
        raise RuntimeError(
            f"ESC-50 metadata references {len(missing_paths)} missing audio files. "
            f"First missing paths:\n{examples}"
        )
    if len(records) != 2000:
        raise RuntimeError(f"Expected 2000 ESC-50 clips, parsed {len(records)} from {csv_path}.")
    if sorted(target_to_category) != list(range(50)):
        raise RuntimeError(
            f"Expected ESC-50 targets 0..49, found {sorted(target_to_category)}."
        )

    class_names = [target_to_category[idx] for idx in range(50)]
    print("Parsed 2000 ESC-50 clips (50 classes, 5 folds).")
    return records, class_names


def parse_speech_commands_dataset(data_dir, sample_rate=16000, clip_seconds=1.0):
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Speech Commands directory not found: {data_dir}")

    class_names = list(SPEECH_COMMANDS_CLASS_NAMES)
    class_map = {name: idx for idx, name in enumerate(class_names)}
    validation_paths = _read_path_list(os.path.join(data_dir, "validation_list.txt"))
    testing_paths = _read_path_list(os.path.join(data_dir, "testing_list.txt"))

    records = []
    missing_paths = []
    wav_paths = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d != SPEECH_COMMANDS_BACKGROUND_NOISE]
        for filename in files:
            if filename.lower().endswith(".wav"):
                wav_paths.append(os.path.join(root, filename))
    available_rel_paths = {
        os.path.relpath(path, data_dir).replace("\\", "/")
        for path in wav_paths
    }
    missing_split_paths = sorted(
        path
        for path in (validation_paths | testing_paths)
        if path not in available_rel_paths and not path.startswith(f"{SPEECH_COMMANDS_BACKGROUND_NOISE}/")
    )
    if missing_split_paths:
        examples = "\n".join(missing_split_paths[:20])
        raise RuntimeError(
            f"Speech Commands split lists reference {len(missing_split_paths)} missing WAV files. "
            f"First missing paths:\n{examples}"
        )

    for path in sorted(wav_paths):
        rel = os.path.relpath(path, data_dir).replace("\\", "/")
        word = rel.split("/", 1)[0].lower()
        if word == SPEECH_COMMANDS_BACKGROUND_NOISE:
            continue
        if not os.path.exists(path):
            missing_paths.append(path)
            continue
        split = "validation" if rel in validation_paths else "test" if rel in testing_paths else "train"
        label_name = word if word in SPEECH_COMMANDS_WORDS else SPEECH_COMMANDS_UNKNOWN
        label = class_map[label_name]
        speaker_id = _speech_speaker_id(os.path.basename(path))
        records.append({
            "path": path,
            "slice_file_name": rel,
            "label": label,
            "fold": split,
            "split": split,
            "cls_name": label_name,
            "raw_word": word,
            "fsID": speaker_id,
            "classID": label,
            "clip_id": rel,
            "duration": clip_seconds,
        })

    noise_dir = os.path.join(data_dir, SPEECH_COMMANDS_BACKGROUND_NOISE)
    if not os.path.isdir(noise_dir):
        raise RuntimeError(
            f"Speech Commands 12-label contract requires {SPEECH_COMMANDS_BACKGROUND_NOISE}: {noise_dir}"
        )
    window_len = int(sample_rate * clip_seconds)
    silence_hop = max(1, window_len // 2)
    for filename in sorted(os.listdir(noise_dir)):
        if not filename.lower().endswith(".wav"):
            continue
        path = os.path.join(noise_dir, filename)
        try:
            info = torchaudio.info(path)
            source_sr = int(info.sample_rate or sample_rate)
            target_num_frames = int(float(info.num_frames) * sample_rate / max(source_sr, 1))
            starts = list(range(0, max(1, target_num_frames - window_len), silence_hop)) or [0]
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect background-noise file: {path}") from exc

        split = "validation" if filename == "running_tap.wav" else "train"
        for idx, start in enumerate(starts):
            rel = os.path.relpath(path, data_dir).replace("\\", "/")
            clip_id = f"{rel}#silence_{idx:04d}_{start}"
            records.append({
                "path": path,
                "slice_file_name": clip_id,
                "label": class_map[SPEECH_COMMANDS_SILENCE],
                "fold": split,
                "split": split,
                "cls_name": SPEECH_COMMANDS_SILENCE,
                "raw_word": SPEECH_COMMANDS_BACKGROUND_NOISE,
                "fsID": f"{filename}",
                "classID": class_map[SPEECH_COMMANDS_SILENCE],
                "clip_id": clip_id,
                "frame_start": start,
                "duration": clip_seconds,
                "cache_full_waveform": True,
            })

    if missing_paths:
        examples = "\n".join(missing_paths[:20])
        raise RuntimeError(
            f"Speech Commands metadata references {len(missing_paths)} missing audio files. "
            f"First missing paths:\n{examples}"
        )
    if not records:
        raise RuntimeError(f"No Speech Commands WAV files found under {data_dir}.")

    split_counts = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    print(f"Parsed {len(records)} Speech Commands records with split counts {split_counts}.")
    return records, class_names


def parse_audio_dataset(dataset_name, data_dir, class_names=None, sample_rate=16000, clip_seconds=None):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "urbansound8k":
        names = list(class_names or URBANSOUND8K_CLASS_NAMES)
        csv_path = os.path.join(data_dir, "metadata", "UrbanSound8K.csv")
        audio_base = os.path.join(data_dir, "audio")
        return parse_dataset(csv_path, audio_base, names), names, dataset_name
    if dataset_name == "esc50":
        records, names = parse_esc50_dataset(data_dir)
        return records, names, dataset_name
    if dataset_name == "speech_commands":
        records, names = parse_speech_commands_dataset(
            data_dir,
            sample_rate=int(sample_rate),
            clip_seconds=float(clip_seconds if clip_seconds is not None else 1.0),
        )
        return records, names, dataset_name
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Use 'urbansound8k', 'esc50', or 'speech_commands'."
    )

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
        base_frame_start = int(r.get("frame_start", 0))
        for i in range(frames_per_clip):
            relative_frame_start = i * frame_hop
            if duration_samples is not None and relative_frame_start >= duration_samples:
                continue
            frame_start = base_frame_start + relative_frame_start
            frame_records.append({
                "path": r["path"],
                "label": r["label"],
                "fold": r["fold"],
                "frame_start": frame_start,
                "slice_file_name": r.get("slice_file_name"),
                "fsID": r.get("fsID"),
                "classID": r.get("classID"),
                "clip_id": r.get("clip_id", r["path"]),
            })
    return frame_records

def load_audio_to_ram(path, sample_rate=16000, clip_seconds=4.0):
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

    if clip_seconds is not None:
        target_len = int(sample_rate * float(clip_seconds))
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]), mode='constant')
        else:
            waveform = waveform[:, :target_len]

    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"Preprocessed waveform contains NaN or Inf: {path}")
    if float(waveform.abs().max().item()) == 0.0:
        raise RuntimeError(f"Preprocessed waveform is completely silent: {path}")

    return path, waveform.contiguous().numpy()  # Convert to numpy array to prevent PyTorch shared memory leaks
