import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

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


def default_data_dir():
    candidates = [
        REPO_ROOT / "data" / "raw" / "UrbanSound8K",
        REPO_ROOT.parents[2] / "data" / "UrbanSound8K",
    ]
    for candidate in candidates:
        if (candidate / "metadata" / "UrbanSound8K.csv").exists():
            return candidate
    return candidates[0]


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


def parse_classes(value):
    if value.strip().lower() == "all":
        return list(range(len(CLASS_NAMES)))
    class_ids = []
    by_name = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            class_id = int(part)
        else:
            class_id = by_name.get(part)
            if class_id is None:
                raise ValueError(f"unknown class '{part}'")
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"class id must be in [0, 9], got {class_id}")
        if class_id not in class_ids:
            class_ids.append(class_id)
    return class_ids


def fmt_pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.2f}%"


def load_metadata(data_dir):
    metadata_path = Path(data_dir) / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    by_file = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_file[row["slice_file_name"]] = {
                "fsID": str(row["fsID"]),
                "classID": int(row["classID"]),
                "class": row["class"],
                "fold": int(row["fold"]),
                "salience": int(row["salience"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
            }
    return by_file


def read_predictions(exp_name, fold, prediction_set):
    pred_path = REPO_ROOT / "experiments" / exp_name / f"fold_{fold}" / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing predictions file: {pred_path}")
    with pred_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if prediction_set not in data:
        raise KeyError(f"{pred_path} does not contain prediction set '{prediction_set}'")
    return data[prediction_set], pred_path


def summarize_fold(exp_name, fold, metadata, class_ids, prediction_set, top_groups):
    predictions, pred_path = read_predictions(exp_name, fold, prediction_set)
    class_rows = []

    for class_id in class_ids:
        groups = collections.defaultdict(
            lambda: {
                "support": 0,
                "correct": 0,
                "confusions": collections.Counter(),
                "files": [],
            }
        )

        for item in predictions:
            if int(item["label"]) != class_id:
                continue
            file_name = Path(item["path"]).name
            meta = metadata.get(file_name)
            fsid = meta["fsID"] if meta else file_name.split("-")[0]
            row = groups[fsid]
            row["support"] += 1
            row["files"].append(file_name)
            predicted = int(item["predicted"])
            if predicted == class_id:
                row["correct"] += 1
            else:
                row["confusions"][CLASS_NAMES[predicted]] += 1

        group_rows = []
        total = sum(row["support"] for row in groups.values())
        total_correct = sum(row["correct"] for row in groups.values())
        for fsid, row in groups.items():
            support = row["support"]
            correct = row["correct"]
            accuracy = correct / support if support else None
            group_rows.append(
                {
                    "fsID": fsid,
                    "support": support,
                    "correct": correct,
                    "accuracy_pct": 100.0 * accuracy if accuracy is not None else None,
                    "top_confusions": [
                        {"predicted": name, "count": count}
                        for name, count in row["confusions"].most_common(5)
                    ],
                    "example_files": row["files"][:5],
                }
            )

        group_rows.sort(
            key=lambda row: (
                row["accuracy_pct"] if row["accuracy_pct"] is not None else 101.0,
                -row["support"],
                row["fsID"],
            )
        )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "support": total,
                "correct": total_correct,
                "accuracy_pct": 100.0 * total_correct / total if total else None,
                "groups": group_rows[:top_groups],
            }
        )

    return {
        "exp_name": exp_name,
        "fold": fold,
        "prediction_set": prediction_set,
        "predictions_path": str(pred_path),
        "classes": class_rows,
    }


def build_report(exp_name, folds, metadata, class_ids, prediction_set, top_groups):
    return {
        "exp_name": exp_name,
        "folds": folds,
        "prediction_set": prediction_set,
        "classes": [CLASS_NAMES[class_id] for class_id in class_ids],
        "fold_results": [
            summarize_fold(exp_name, fold, metadata, class_ids, prediction_set, top_groups)
            for fold in folds
        ],
    }


def write_markdown(report, path):
    lines = []
    lines.append(f"# Source Confusion Analysis: {report['exp_name']}")
    lines.append("")
    lines.append(f"Prediction set: `{report['prediction_set']}`")
    lines.append("")
    lines.append(f"Folds: {', '.join(str(fold) for fold in report['folds'])}")
    lines.append("")

    for fold_result in report["fold_results"]:
        lines.append(f"## Fold {fold_result['fold']}")
        lines.append("")
        for class_result in fold_result["classes"]:
            lines.append(
                f"### {class_result['class_name']} "
                f"({class_result['correct']}/{class_result['support']}, "
                f"{fmt_pct(class_result['accuracy_pct'])})"
            )
            lines.append("")
            lines.append("| fsID | Accuracy | Correct | Support | Main confusions | Example files |")
            lines.append("|---|---:|---:|---:|---|---|")
            for group in class_result["groups"]:
                confusions = ", ".join(
                    f"{item['predicted']} {item['count']}" for item in group["top_confusions"]
                ) or "none"
                examples = ", ".join(group["example_files"])
                lines.append(
                    f"| {group['fsID']} | {fmt_pct(group['accuracy_pct'])} | "
                    f"{group['correct']} | {group['support']} | {confusions} | {examples} |"
                )
            lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize UrbanSound8K prediction errors by fsID source group."
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--folds", default="1-3")
    parser.add_argument("--classes", default="all", help="Class names or ids, comma separated, or all.")
    parser.add_argument("--prediction_set", default="last_snapshot_predictions")
    parser.add_argument("--top_groups", type=int, default=8)
    parser.add_argument("--data_dir", default=str(default_data_dir()))
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    folds = parse_folds(args.folds)
    class_ids = parse_classes(args.classes)
    metadata = load_metadata(args.data_dir)
    report = build_report(args.exp_name, folds, metadata, class_ids, args.prediction_set, args.top_groups)

    exp_root = REPO_ROOT / "experiments" / args.exp_name
    output_json = Path(args.output_json) if args.output_json else exp_root / "source_confusions.json"
    output_md = Path(args.output_md) if args.output_md else exp_root / "source_confusions.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_md)

    print(f"Source confusion JSON: {output_json}")
    print(f"Source confusion report: {output_md}")


if __name__ == "__main__":
    main()
