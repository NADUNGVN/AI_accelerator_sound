#!/usr/bin/env python3
"""Aggregate fold metrics for a multi-fold experiment.

Usage:
  python scripts/summarize_10fold.py --exp_name paper_abdoli_gamma
  python scripts/summarize_10fold.py --exp_dir experiments/paper_abdoli_gamma
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


def load_fold_metrics(exp_dir: Path):
    rows = []
    for fold in range(1, 11):
        path = exp_dir / f"fold_{fold}" / "metrics.json"
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "fold": fold,
            "path": str(path),
            "protocol": m.get("protocol"),
            "best_val_clip_acc": m.get("best_val_clip_acc"),
            "test_acc_best_val_model": m.get("test_acc_best_val_model"),
            "test_acc_last_snapshot": m.get("test_acc_last_snapshot"),
            "final_epoch": m.get("final_epoch"),
            "stopped_early": m.get("stopped_early"),
            "train_clip_count": m.get("train_clip_count"),
            "val_clip_count": m.get("val_clip_count"),
            "test_clip_count": m.get("test_clip_count"),
            "frame_length": m.get("frame_length"),
            "frame_hop": m.get("frame_hop"),
            "variant": m.get("variant"),
            "loss_type": m.get("loss_type"),
            "optimizer": m.get("optimizer"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize 10-fold experiment metrics")
    parser.add_argument("--exp_name", type=str, default="paper_abdoli_gamma")
    parser.add_argument("--exp_dir", type=str, default=None, help="Override experiments/<exp_name>")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exp_dir = Path(args.exp_dir) if args.exp_dir else root / "experiments" / args.exp_name
    if not exp_dir.is_dir():
        print(f"Experiment dir not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    rows = load_fold_metrics(exp_dir)
    if not rows:
        print(f"No fold metrics found under {exp_dir}/fold_*/metrics.json")
        sys.exit(1)

    print(f"Experiment: {exp_dir}")
    print(f"Folds completed: {len(rows)}/10")
    print()
    print(f"{'fold':>4}  {'best_val%':>10}  {'test_best%':>10}  {'test_last%':>10}  {'epochs':>6}  early")
    print("-" * 62)

    best_vals, test_bests, test_lasts = [], [], []
    for r in rows:
        bv = r["best_val_clip_acc"]
        tb = r["test_acc_best_val_model"]
        tl = r["test_acc_last_snapshot"]
        if bv is not None:
            best_vals.append(float(bv))
        if tb is not None:
            test_bests.append(float(tb))
        if tl is not None:
            test_lasts.append(float(tl))

        def pct(x):
            return f"{100.0 * x:8.2f}" if x is not None else "     n/a"

        print(
            f"{r['fold']:4d}  {pct(bv)}  {pct(tb)}  {pct(tl)}  "
            f"{str(r['final_epoch'] or '-'):>6}  {r['stopped_early']}"
        )

    print("-" * 62)

    def report(name, vals):
        m, s = mean_std(vals)
        if m is None:
            print(f"{name}: n/a")
        else:
            print(f"{name}: {100.0 * m:.2f}% ± {100.0 * s:.2f}%  (n={len(vals)})")

    print()
    report("Mean best val clip acc", best_vals)
    report("Mean test (best-val model)", test_bests)
    report("Mean test (last model)", test_lasts)
    print()
    print("Paper Abdoli claim (Table 3/4 Gammatone): ~89% mean accuracy")
    print("Primary comparable metric: Mean test (best-val model) under clean_8_1_1")

    summary = {
        "exp_dir": str(exp_dir),
        "n_folds": len(rows),
        "folds": rows,
        "mean_best_val_clip_acc": mean_std(best_vals)[0],
        "std_best_val_clip_acc": mean_std(best_vals)[1],
        "mean_test_acc_best_val_model": mean_std(test_bests)[0],
        "std_test_acc_best_val_model": mean_std(test_bests)[1],
        "mean_test_acc_last": mean_std(test_lasts)[0],
        "std_test_acc_last": mean_std(test_lasts)[1],
    }

    out_path = Path(args.out) if args.out else exp_dir / "summary_10fold.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
