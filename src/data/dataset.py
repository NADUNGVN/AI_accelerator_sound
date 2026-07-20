"""UrbanSound8K loading, framing, and RAM-cached frame datasets.

Paper-faithful framing (Abdoli et al. 2019):
  * Resample to 16 kHz mono.
  * Frame on the **actual** waveform length (no forced 4 s zero canvas).
  * Only pad a frame when the whole clip is shorter than ``frame_length``.
  * Optionally drop near-silent frames so zero-padded / silent segments
    are not trained as class labels (TCAM-era contamination).
"""

from __future__ import annotations

import os
import csv
from typing import Any

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


# Peak |x| below this is treated as near-silent (float32 waveforms ~[-1, 1]).
DEFAULT_ZERO_ABS_THRESHOLD = 1e-4


class CachedUrbanSoundFrameDataset(Dataset):
    """Frame-level dataset reading preloaded waveforms from a RAM dict."""

    def __init__(self, records, cached_waveforms, frame_length=16000):
        self.records = records
        self.cached_waveforms = cached_waveforms
        self.frame_length = int(frame_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = record["path"]
        label = record["label"]
        frame_start = int(record["frame_start"])

        waveform_np = self.cached_waveforms[path]
        frame_np = waveform_np[:, frame_start:frame_start + self.frame_length]
        frame = torch.from_numpy(np.ascontiguousarray(frame_np))

        # Only needed when the whole clip is shorter than frame_length.
        if frame.shape[-1] < self.frame_length:
            padding = self.frame_length - frame.shape[-1]
            frame = F.pad(frame, (0, padding), mode="constant")

        return frame, label


def parse_dataset(csv_path, audio_base_dir, class_names):
    """
    Parse UrbanSound8K CSV → standard 10-class clip records.
    Filters optional custom class ``rail_vehicle``. Fails fast on missing files.
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

            if cls_name == "rail_vehicle":
                continue

            if cls_name not in class_map:
                unknown_classes.append(cls_name)
                continue

            path = os.path.join(audio_base_dir, f"fold{fold}", filename)
            if not os.path.exists(path):
                missing_paths.append(path)
                continue

            record = {
                "path": path,
                "slice_file_name": filename,
                "label": class_map[cls_name],
                "fold": fold,
                "cls_name": cls_name,
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
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
    ] else None
    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} UrbanSound8K 10-class clips after filtering, "
            f"but parsed {len(records)} clips from {csv_path}."
        )

    print(f"Parsed {len(records)} standard audio clips (10 classes, rail_vehicle excluded).")
    return records


def load_audio_to_ram(
    path,
    sample_rate=16000,
    pad_to_seconds=None,
    max_seconds=4.0,
):
    """
    Load one audio file → mono float32 numpy array (1, T).

    Parameters
    ----------
    pad_to_seconds:
        If set (e.g. 4.0), zero-pad / truncate every clip to exactly that length
        (legacy TCAM behaviour). Paper Abdoli path uses ``None`` = keep real length.
    max_seconds:
        Hard truncate longer than this (US8K max is 4 s). ``None`` disables.
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

    if max_seconds is not None:
        max_len = int(sample_rate * max_seconds)
        if waveform.shape[-1] > max_len:
            waveform = waveform[:, :max_len]

    if pad_to_seconds is not None:
        target_len = int(sample_rate * pad_to_seconds)
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]), mode="constant")
        else:
            waveform = waveform[:, :target_len]

    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"Preprocessed waveform contains NaN or Inf: {path}")
    if float(waveform.abs().max().item()) == 0.0:
        raise RuntimeError(f"Preprocessed waveform is completely silent: {path}")

    return path, waveform.contiguous().numpy()


def frame_offsets_for_length(num_samples: int, frame_length: int, frame_hop: int) -> list[int]:
    """Sliding-window start indices over a clip of ``num_samples`` samples."""
    frame_length = int(frame_length)
    frame_hop = int(frame_hop)
    if num_samples <= 0:
        return []
    if num_samples < frame_length:
        # Single frame; caller will right-pad to frame_length.
        return [0]
    offsets = list(range(0, num_samples - frame_length + 1, frame_hop))
    # Cover the clip tail when length is not a multiple of hop.
    last_start = num_samples - frame_length
    if offsets and last_start > offsets[-1]:
        offsets.append(last_start)
    elif not offsets:
        offsets = [0]
    return offsets


def frame_peak_abs(waveform_np: np.ndarray, start: int, frame_length: int) -> float:
    """Peak absolute amplitude of waveform_np[:, start:start+frame_length]."""
    end = min(start + frame_length, waveform_np.shape[-1])
    if end <= start:
        return 0.0
    segment = waveform_np[:, start:end]
    return float(np.max(np.abs(segment))) if segment.size else 0.0


def generate_frame_records(
    clip_records,
    cached_waveforms,
    frame_length=16000,
    frame_hop=8000,
    skip_near_zero=True,
    zero_abs_threshold=DEFAULT_ZERO_ABS_THRESHOLD,
):
    """
    Expand clip records into frame records using each clip's **real** length.

    Returns
    -------
    frame_records : list[dict]
    stats : dict
        candidate / kept / skipped_zero / short_clips / fallback_energy counts.
    """
    frame_length = int(frame_length)
    frame_hop = int(frame_hop)
    stats: dict[str, Any] = {
        "candidate_frames": 0,
        "kept_frames": 0,
        "skipped_near_zero": 0,
        "short_clips": 0,
        "fallback_energy_clips": 0,
        "clips": len(clip_records),
    }

    frame_records = []
    for r in clip_records:
        path = r["path"]
        if path not in cached_waveforms:
            raise KeyError(f"Waveform not in cache for path: {path}")
        w = cached_waveforms[path]
        T = int(w.shape[-1])
        if T <= 0:
            continue

        if T < frame_length:
            stats["short_clips"] += 1

        offsets = frame_offsets_for_length(T, frame_length, frame_hop)
        kept_starts: list[int] = []

        for start in offsets:
            stats["candidate_frames"] += 1
            peak = frame_peak_abs(w, start, frame_length)
            if skip_near_zero and peak < zero_abs_threshold:
                stats["skipped_near_zero"] += 1
                continue
            kept_starts.append(start)

        if not kept_starts:
            # Never drop a clip entirely: keep the highest-energy candidate frame.
            best_start, best_peak = offsets[0], -1.0
            for start in offsets:
                peak = frame_peak_abs(w, start, frame_length)
                if peak > best_peak:
                    best_peak, best_start = peak, start
            kept_starts = [best_start]
            stats["fallback_energy_clips"] += 1

        for start in kept_starts:
            stats["kept_frames"] += 1
            frame_records.append({
                "path": path,
                "label": r["label"],
                "fold": r["fold"],
                "frame_start": int(start),
            })

    stats["skip_rate"] = (
        stats["skipped_near_zero"] / stats["candidate_frames"]
        if stats["candidate_frames"] else 0.0
    )
    return frame_records, stats


def extract_clip_frame_tensors(
    waveform_np: np.ndarray,
    frame_length: int,
    frame_hop: int,
    skip_near_zero: bool = True,
    zero_abs_threshold: float = DEFAULT_ZERO_ABS_THRESHOLD,
) -> list[torch.Tensor]:
    """
    Build a list of (1, frame_length) tensors for one clip (eval-time helper).
    Uses real length; right-pads only if the clip is shorter than frame_length.
    """
    frame_length = int(frame_length)
    frame_hop = int(frame_hop)
    T = int(waveform_np.shape[-1])
    offsets = frame_offsets_for_length(T, frame_length, frame_hop)

    frames: list[torch.Tensor] = []
    peaks: list[float] = []
    for start in offsets:
        peak = frame_peak_abs(waveform_np, start, frame_length)
        peaks.append(peak)
        if skip_near_zero and peak < zero_abs_threshold:
            continue
        end = min(start + frame_length, T)
        frame = torch.from_numpy(np.ascontiguousarray(waveform_np[:, start:end]))
        if frame.shape[-1] < frame_length:
            frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode="constant")
        frames.append(frame)

    if not frames:
        # Fallback: highest-energy offset
        best_i = int(np.argmax(peaks)) if peaks else 0
        start = offsets[best_i] if offsets else 0
        end = min(start + frame_length, T)
        frame = torch.from_numpy(np.ascontiguousarray(waveform_np[:, start:end]))
        if frame.shape[-1] < frame_length:
            frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode="constant")
        frames = [frame]

    return frames


def summarize_length_stats(cached_waveforms, sample_rate=16000) -> dict[str, Any]:
    """Duration histogram helpers for logging."""
    lengths = [int(w.shape[-1]) for w in cached_waveforms.values()]
    if not lengths:
        return {"n": 0}
    arr = np.asarray(lengths, dtype=np.float64)
    secs = arr / float(sample_rate)
    return {
        "n": len(lengths),
        "samples_min": int(arr.min()),
        "samples_max": int(arr.max()),
        "samples_mean": float(arr.mean()),
        "seconds_min": float(secs.min()),
        "seconds_max": float(secs.max()),
        "seconds_mean": float(secs.mean()),
        "n_shorter_than_1s": int((secs < 1.0).sum()),
        "n_shorter_than_2s": int((secs < 2.0).sum()),
        "n_exactly_max": int((arr == arr.max()).sum()),
    }
