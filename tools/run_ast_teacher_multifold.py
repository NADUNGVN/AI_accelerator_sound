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


def run_streamed(command, log_path, dry_run=False):
    print("\n$", " ".join(str(part) for part in command))
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def per_class_map(payload):
    rows = (((payload or {}).get("test") or {}).get("per_class") or [])
    return {row["class_name"]: pct(row.get("accuracy")) for row in rows}


def collect_fold(exp_root, fold):
    fold_dir = Path(exp_root) / f"fold_{fold}"
    metrics = read_json(fold_dir / "metrics.json")
    if metrics is None:
        return {
            "fold": fold,
            "exp_dir": str(fold_dir),
            "has_metrics": False,
            "completed_epochs": None,
            "planned_epochs": None,
            "early_stopped": None,
            "best_epoch": None,
            "best_val_acc_pct": None,
            "best_test_acc_pct": None,
            "final_val_acc_pct": None,
            "final_test_acc_pct": None,
            "final_worst_class": None,
            "final_worst_class_acc_pct": None,
            "best_per_class_pct": {},
            "final_per_class_pct": {},
        }

    best = metrics.get("best") or {}
    final = metrics.get("final") or {}
    final_per_class = per_class_map(final)
    if final_per_class:
        final_worst_class = min(final_per_class, key=final_per_class.get)
        final_worst_class_acc = final_per_class[final_worst_class]
    else:
        final_worst_class = None
        final_worst_class_acc = None

    return {
        "fold": fold,
        "exp_dir": str(fold_dir),
        "has_metrics": True,
        "protocol": metrics.get("protocol"),
        "seed": metrics.get("seed"),
        "model_name": metrics.get("model_name"),
        "counts": metrics.get("counts"),
        "source_label_overlap": metrics.get("source_label_overlap"),
        "params_total": ((metrics.get("params") or {}).get("total")),
        "completed_epochs": metrics.get("completed_epochs"),
        "planned_epochs": metrics.get("planned_epochs"),
        "early_stopped": metrics.get("early_stopped"),
        "best_epoch": best.get("epoch"),
        "best_val_acc_pct": pct(((best.get("val") or {}).get("accuracy"))),
        "best_test_acc_pct": pct(((best.get("test") or {}).get("accuracy"))),
        "final_val_acc_pct": pct(((final.get("val") or {}).get("accuracy"))),
        "final_test_acc_pct": pct(((final.get("test") or {}).get("accuracy"))),
        "final_worst_class": final_worst_class,
        "final_worst_class_acc_pct": final_worst_class_acc,
        "best_per_class_pct": per_class_map(best),
        "final_per_class_pct": final_per_class,
    }


def build_summary(exp_root, folds):
    rows = [collect_fold(exp_root, fold) for fold in folds]
    metric_keys = [
        "best_val_acc_pct",
        "best_test_acc_pct",
        "final_val_acc_pct",
        "final_test_acc_pct",
        "final_worst_class_acc_pct",
    ]
    aggregate = {}
    completed_rows = [row for row in rows if row.get("has_metrics")]
    for key in metric_keys:
        mean, std = mean_std([row.get(key) for row in completed_rows])
        aggregate[key] = {"mean": mean, "std": std}

    per_class_final = {}
    per_class_best = {}
    for class_name in CLASS_NAMES:
        mean, std = mean_std([row["final_per_class_pct"].get(class_name) for row in completed_rows])
        per_class_final[class_name] = {"mean": mean, "std": std}
        mean, std = mean_std([row["best_per_class_pct"].get(class_name) for row in completed_rows])
        per_class_best[class_name] = {"mean": mean, "std": std}

    return {
        "exp_root": str(exp_root),
        "folds": folds,
        "completed_folds": [row["fold"] for row in completed_rows],
        "rows": rows,
        "aggregate": aggregate,
        "per_class_best_test": per_class_best,
        "per_class_final_test": per_class_final,
    }


def write_summary_files(summary, exp_root):
    exp_root = Path(exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    json_path = exp_root / "multifold_summary.json"
    md_path = exp_root / "multifold_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"# AST Teacher Multifold Summary: {exp_root.name}",
        "",
        f"Requested folds: {', '.join(str(f) for f in summary['folds'])}",
        f"Completed folds: {', '.join(str(f) for f in summary['completed_folds']) or 'none'}",
        "",
        "## Fold Results",
        "",
        "| Fold | Epochs | Early stop | Best epoch | Best val | Best-val test | Final val | Final test | Worst final class | Worst final acc |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary["rows"]:
        planned_epochs = row.get("planned_epochs")
        completed_epochs = row.get("completed_epochs")
        epoch_text = "N/A" if planned_epochs is None else f"{completed_epochs}/{planned_epochs}"
        lines.append(
            "| "
            f"{row['fold']} | "
            f"{epoch_text} | "
            f"{'yes' if row.get('early_stopped') else 'no'} | "
            f"{row.get('best_epoch') or 'N/A'} | "
            f"{fmt_pct(row.get('best_val_acc_pct'))} | "
            f"{fmt_pct(row.get('best_test_acc_pct'))} | "
            f"{fmt_pct(row.get('final_val_acc_pct'))} | "
            f"{fmt_pct(row.get('final_test_acc_pct'))} | "
            f"{row.get('final_worst_class') or 'N/A'} | "
            f"{fmt_pct(row.get('final_worst_class_acc_pct'))} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Metric | Mean | Std |",
            "|---|---:|---:|",
        ]
    )
    labels = {
        "best_val_acc_pct": "Best validation",
        "best_test_acc_pct": "Validation-selected test",
        "final_val_acc_pct": "Final validation",
        "final_test_acc_pct": "Final test",
        "final_worst_class_acc_pct": "Worst final class",
    }
    for key, label in labels.items():
        item = summary["aggregate"][key]
        lines.append(f"| {label} | {fmt_pct(item['mean'])} | {fmt_pct(item['std'])} |")

    lines.extend(
        [
            "",
            "## Best-Validation Test Per Class",
            "",
            "| Class | Mean | Std |",
            "|---|---:|---:|",
        ]
    )
    for class_name, item in summary["per_class_best_test"].items():
        lines.append(f"| {class_name} | {fmt_pct(item['mean'])} | {fmt_pct(item['std'])} |")

    lines.extend(
        [
            "",
            "## Final Test Per Class",
            "",
            "| Class | Mean | Std |",
            "|---|---:|---:|",
        ]
    )
    for class_name, item in summary["per_class_final_test"].items():
        lines.append(f"| {class_name} | {fmt_pct(item['mean'])} | {fmt_pct(item['std'])} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run and summarize AST teacher source-safe multifold training.")
    parser.add_argument("--exp_name", default="local_ast_teacher_full10_12ep")
    parser.add_argument("--folds", default="1-10", help="Comma/range list, e.g. 1,2,3 or 1-10.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--early_stop_warmup", type=int, default=6)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hf_cache_dir", default="experiments/smoke_ast_embedding_probe/hf_cache")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--eval_test_each_epoch", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    folds = parse_folds(args.folds)
    exp_root = repo_root / "experiments" / args.exp_name
    log_root = repo_root / "logs" / args.exp_name

    for fold in folds:
        fold_dir = exp_root / f"fold_{fold}"
        metrics_path = fold_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"\nSkipping fold {fold}: existing {metrics_path}")
            continue

        command = [
            args.python,
            str(repo_root / "tools" / "finetune_ast_teacher.py"),
            "--exp_name",
            args.exp_name,
            "--fold",
            str(fold),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--accum_steps",
            str(args.accum_steps),
            "--encoder_lr",
            str(args.encoder_lr),
            "--head_lr",
            str(args.head_lr),
            "--early_stop_warmup",
            str(args.early_stop_warmup),
            "--early_stop_patience",
            str(args.early_stop_patience),
            "--num_workers",
            str(args.num_workers),
            "--hf_cache_dir",
            args.hf_cache_dir,
        ]
        if args.data_dir is not None:
            command.extend(["--data_dir", args.data_dir])
        if args.local_files_only:
            command.append("--local_files_only")
        if args.eval_test_each_epoch:
            command.append("--eval_test_each_epoch")

        run_streamed(command, log_root / f"fold_{fold}.log", dry_run=args.dry_run)

    if not args.dry_run:
        summary = build_summary(exp_root, folds)
        json_path, md_path = write_summary_files(summary, exp_root)
        print(f"\nSummary written: {json_path}")
        print(f"Summary report : {md_path}")


if __name__ == "__main__":
    main()
