#!/usr/bin/env python
"""Verify SDP 8-1-1 split fingerprint matches student contract (seed 83).

Expect fold1: train=6996, val=866, test=870, train/test overlap=0.

Does NOT import sklearn (safe for minimal teacher env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import parse_dataset  # noqa: E402
from train import (  # noqa: E402
    make_stratified_source_group_train_val_test_split,
    source_label_overlap_summary,
)

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

EXPECTED_F1 = {"train": 6996, "val": 866, "test": 870}


def default_data_dir() -> str:
    candidates = [
        REPO / "data" / "UrbanSound8K",
        REPO / "data" / "raw" / "UrbanSound8K",
    ]
    for c in candidates:
        if (c / "metadata" / "UrbanSound8K.csv").exists():
            return str(c)
    return str(candidates[0])


def fingerprint(keys: list[str]) -> str:
    h = hashlib.sha1()
    for p in sorted(keys):
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def row_key(r) -> str:
    for k in ("path", "filepath", "file", "slice_file_name", "filename"):
        if k in r and r[k]:
            return str(r[k])
    return json.dumps(r, sort_keys=True, default=str)


def make_split(clip_records, fold, protocol, seed):
    protocol = protocol.lower()
    if protocol == "source_group_8_1_1":
        train_clips, val_clips, test_records, val_bucket = make_stratified_source_group_train_val_test_split(
            clip_records,
            test_bucket=fold,
            seed=seed,
        )
        return {
            "fold": fold,
            "protocol": protocol,
            "train": train_clips,
            "val": val_clips,
            "test": test_records,
            "uses_validation": True,
            "val_bucket": val_bucket,
        }
    raise ValueError(f"This verifier only supports source_group_8_1_1, got {protocol}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=default_data_dir())
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--protocol", default="source_group_8_1_1")
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")
    if not os.path.isfile(csv_path):
        print(f"[FAIL] missing metadata: {csv_path}")
        sys.exit(2)

    records = parse_dataset(csv_path, audio_base, CLASS_NAMES)
    split = make_split(records, args.fold, args.protocol, seed=args.seed)
    n_train, n_val, n_test = len(split["train"]), len(split["val"]), len(split["test"])
    ov_tt = source_label_overlap_summary(split["train"], split["test"])
    ov_tv = source_label_overlap_summary(split["train"], split["val"])

    report = {
        "data_dir": args.data_dir,
        "protocol": args.protocol,
        "seed": args.seed,
        "fold": args.fold,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "overlap_train_test": ov_tt.get("count"),
        "overlap_train_val": ov_tv.get("count"),
        "fingerprints": {
            "train": fingerprint([row_key(r) for r in split["train"]]),
            "val": fingerprint([row_key(r) for r in split["val"]]),
            "test": fingerprint([row_key(r) for r in split["test"]]),
        },
    }
    print(json.dumps(report, indent=2))

    ok = True
    if args.fold == 1 and args.protocol == "source_group_8_1_1" and args.seed == 83:
        for split_name, exp in EXPECTED_F1.items():
            got = report["counts"][split_name]
            if got != exp:
                print(f"[FAIL] {split_name} count {got} != expected {exp}")
                ok = False
        if report["overlap_train_test"] != 0:
            print("[FAIL] train/test source-label overlap != 0")
            ok = False
        if report["overlap_train_val"] != 0:
            print("[FAIL] train/val source-label overlap != 0")
            ok = False

    if ok:
        print("[OK] SDP fingerprint matches student contract.")
        sys.exit(0)
    print("[FAIL] fix data_dir or split code before training teacher.")
    sys.exit(1)


if __name__ == "__main__":
    main()
