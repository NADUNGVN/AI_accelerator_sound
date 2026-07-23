import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


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


def parse_folds(value):
    folds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    unique = []
    for fold in folds:
        if not 1 <= fold <= 10:
            raise ValueError(f"fold must be in [1, 10], got {fold}")
        if fold not in unique:
            unique.append(fold)
    return unique


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    if value is None:
        return None
    return 100.0 * float(value)


def fmt_pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.2f}%"


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def run_command(command, dry_run=False):
    print("\n$", " ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def final_cycle(analysis):
    if not analysis:
        return None
    cycles = analysis.get("cycles") or []
    for cycle in reversed(cycles):
        if str(cycle.get("cycle")) == "final":
            return cycle
    return cycles[-1] if cycles else None


def collect_fold(exp_root, fold):
    fold_dir = Path(exp_root) / f"fold_{fold}"
    metrics = read_json(fold_dir / "metrics.json")
    analysis = read_json(fold_dir / "analysis_all_cycles.json")
    final = final_cycle(analysis)

    row = {
        "fold": fold,
        "exp_dir": str(fold_dir),
        "has_metrics": metrics is not None,
        "has_analysis": analysis is not None,
        "dataset": metrics.get("dataset") if metrics else None,
        "class_names": metrics.get("class_names") if metrics else None,
        "num_classes": metrics.get("num_classes") if metrics else None,
        "config_path": metrics.get("config_path") if metrics else None,
        "params": (metrics.get("model_params") or {}).get("params_with_bias") if metrics else None,
        "macs_per_clip": metrics.get("model_conv_linear_macs_per_clip_eval") if metrics else None,
        "best_val_acc_pct": pct(metrics.get("best_val_clip_acc")) if metrics else None,
        "best_val_test_acc_pct": pct(metrics.get("test_acc_best_val_model")) if metrics else None,
        "final_test_acc_pct": pct(metrics.get("test_acc_last_snapshot")) if metrics else None,
        "ensemble_test_acc_pct": pct(metrics.get("test_acc_ensemble")) if metrics else None,
        "final_cycle_test_acc_pct": pct(((final or {}).get("test") or {}).get("accuracy")),
        "final_cycle_val_acc_pct": pct(((final or {}).get("val") or {}).get("accuracy")),
        "planned_epochs": metrics.get("epochs") if metrics else None,
        "completed_epochs": metrics.get("completed_epochs") if metrics else None,
        "early_stopped": metrics.get("early_stopped") if metrics else None,
        "final_per_class_pct": {},
    }

    per_class = (((final or {}).get("test") or {}).get("per_class") or [])
    for item in per_class:
        row["final_per_class_pct"][item["class_name"]] = pct(item.get("accuracy"))
    if per_class:
        class_values = list(row["final_per_class_pct"].values())
        row["final_worst_class"] = min(row["final_per_class_pct"], key=row["final_per_class_pct"].get)
        row["final_worst_class_acc_pct"] = min(class_values)
    else:
        row["final_worst_class"] = None
        row["final_worst_class_acc_pct"] = None
    return row


def build_summary(exp_root, folds):
    rows = [collect_fold(exp_root, fold) for fold in folds]
    class_names = next((row["class_names"] for row in rows if row.get("class_names")), CLASS_NAMES)
    dataset = next((row["dataset"] for row in rows if row.get("dataset")), None)
    metric_keys = [
        "best_val_acc_pct",
        "best_val_test_acc_pct",
        "final_test_acc_pct",
        "ensemble_test_acc_pct",
        "final_cycle_test_acc_pct",
        "final_cycle_val_acc_pct",
        "final_worst_class_acc_pct",
    ]
    aggregate = {}
    for key in metric_keys:
        mean, std = mean_std([row.get(key) for row in rows])
        aggregate[key] = {"mean": mean, "std": std}

    per_class = {}
    for class_name in class_names:
        values = [row["final_per_class_pct"].get(class_name) for row in rows]
        mean, std = mean_std(values)
        per_class[class_name] = {"mean": mean, "std": std}

    return {
        "exp_root": str(exp_root),
        "dataset": dataset,
        "class_names": class_names,
        "folds": folds,
        "rows": rows,
        "aggregate": aggregate,
        "per_class_final_test": per_class,
    }


def write_summary_files(summary, exp_root):
    exp_root = Path(exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    json_path = exp_root / "multifold_summary.json"
    md_path = exp_root / "multifold_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append(f"# Multifold Summary: {exp_root.name}")
    lines.append("")
    if summary.get("dataset"):
        lines.append(f"Dataset: {summary['dataset']}")
        lines.append("")
    if summary.get("class_names"):
        lines.append(f"Classes: {len(summary['class_names'])}")
        lines.append("")
    lines.append(f"Folds: {', '.join(str(f) for f in summary['folds'])}")
    lines.append("")
    lines.append("## Fold Results")
    lines.append("")
    lines.append("| Fold | Epochs | Early stop | Best val | Best-val test | Final test | Ensemble | Worst final class | Worst final acc |")
    lines.append("|---:|---:|:---:|---:|---:|---:|---:|---|---:|")
    for row in summary["rows"]:
        planned_epochs = row.get("planned_epochs")
        completed_epochs = row.get("completed_epochs")
        if completed_epochs is None and planned_epochs is not None:
            completed_epochs = planned_epochs
        epoch_text = "N/A" if planned_epochs is None else f"{completed_epochs}/{planned_epochs}"
        lines.append(
            "| "
            f"{row['fold']} | "
            f"{epoch_text} | "
            f"{'yes' if row.get('early_stopped') else 'no'} | "
            f"{fmt_pct(row['best_val_acc_pct'])} | "
            f"{fmt_pct(row['best_val_test_acc_pct'])} | "
            f"{fmt_pct(row['final_test_acc_pct'])} | "
            f"{fmt_pct(row['ensemble_test_acc_pct'])} | "
            f"{row.get('final_worst_class') or 'N/A'} | "
            f"{fmt_pct(row.get('final_worst_class_acc_pct'))} |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---:|---:|")
    labels = {
        "best_val_acc_pct": "Best validation",
        "best_val_test_acc_pct": "Validation-selected test",
        "final_test_acc_pct": "Final test",
        "ensemble_test_acc_pct": "Last-2 ensemble",
        "final_worst_class_acc_pct": "Worst final class",
    }
    for key, label in labels.items():
        item = summary["aggregate"][key]
        lines.append(f"| {label} | {fmt_pct(item['mean'])} | {fmt_pct(item['std'])} |")
    lines.append("")
    lines.append("## Final Test Per Class")
    lines.append("")
    lines.append("| Class | Mean | Std |")
    lines.append("|---|---:|---:|")
    for class_name, item in summary["per_class_final_test"].items():
        lines.append(f"| {class_name} | {fmt_pct(item['mean'])} | {fmt_pct(item['std'])} |")
    lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run and summarize source-group multi-fold experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--folds", default="1-3", help="Comma/range list, e.g. 1,2,3 or 1-5.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run train/analyze.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--max_test_clips", type=int, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--eval_modes", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    folds = parse_folds(args.folds)
    exp_root = repo_root / "experiments" / args.exp_name

    for fold in folds:
        fold_dir = exp_root / f"fold_{fold}"
        metrics_path = fold_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"\nSkipping fold {fold}: existing {metrics_path}")
        else:
            command = [
                args.python,
                str(repo_root / "train.py"),
                "--fold",
                str(fold),
                "--config",
                args.config,
                "--exp_name",
                args.exp_name,
            ]
            if args.epochs is not None:
                command.extend(["--epochs", str(args.epochs)])
            if args.batch_size is not None:
                command.extend(["--batch_size", str(args.batch_size)])
            if args.lr is not None:
                command.extend(["--lr", str(args.lr)])
            if args.data_dir is not None:
                command.extend(["--data_dir", args.data_dir])
            if args.max_train_clips is not None:
                command.extend(["--max_train_clips", str(args.max_train_clips)])
            if args.max_val_clips is not None:
                command.extend(["--max_val_clips", str(args.max_val_clips)])
            if args.max_test_clips is not None:
                command.extend(["--max_test_clips", str(args.max_test_clips)])
            run_command(command, dry_run=args.dry_run)

        if args.analyze:
            analysis_path = fold_dir / "analysis_all_cycles.json"
            if args.skip_existing and analysis_path.exists():
                print(f"\nSkipping fold {fold} analysis: existing {analysis_path}")
            else:
                command = [
                    args.python,
                    str(repo_root / "tools" / "analyze_experiment.py"),
                    "--exp_dir",
                    str(fold_dir),
                    "--fold",
                    str(fold),
                    "--config",
                    args.config,
                    "--eval_all_cycles",
                ]
                if args.eval_modes:
                    command.append("--eval_modes")
                if args.data_dir is not None:
                    command.extend(["--data_dir", args.data_dir])
                run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        summary = build_summary(exp_root, folds)
        json_path, md_path = write_summary_files(summary, exp_root)
        print(f"\nSummary written: {json_path}")
        print(f"Summary report : {md_path}")


if __name__ == "__main__":
    main()
