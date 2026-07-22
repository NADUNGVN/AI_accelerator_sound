#!/usr/bin/env python
"""Verify SDP 8-1-1 split fingerprint matches student contract (seed 83).

Expect fold1: train=6996, val=866, test=870, train/test overlap=0.
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

from tools.source_safe_feature_probe import CLASS_NAMES, make_split  # noqa: E402
from src.data import parse_dataset  # noqa: E402
from train import source_label_overlap_summary  # noqa: E402

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


def fingerprint(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:16]


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

    def paths(rows):
        return [r.get("path") or r.get("filepath") or r.get("file") or str(r) for r in rows]

    # robust path key
    def row_key(r):
        for k in ("path", "filepath", "file", "slice_file_name", "filename"):
            if k in r and r[k]:
                return str(r[k])
        return json.dumps(r, sort_keys=True, default=str)

    train_fp = fingerprint([row_key(r) for r in split["train"]])
    val_fp = fingerprint([row_key(r) for r in split["val"]])
    test_fp = fingerprint([row_key(r) for r in split["test"]])

    report = {
        "data_dir": args.data_dir,
        "protocol": args.protocol,
        "seed": args.seed,
        "fold": args.fold,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "overlap_train_test": ov_tt.get("count"),
        "overlap_train_val": ov_tv.get("count"),
        "fingerprints": {"train": train_fp, "val": val_fp, "test": test_fp},
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
        print("[OK] SDP fingerprint matches student contract (or non-f1 check passed counts).")
        sys.exit(0)
    print("[FAIL] fix data_dir or split code before training teacher.")
    sys.exit(1)


if __name__ == "__main__":
    main()
