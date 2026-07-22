import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_audio_to_ram, parse_dataset
from train import default_data_dir, source_label_overlap_summary
from tools.source_safe_feature_probe import (
    CLASS_NAMES,
    aggregate_results,
    apply_smoke_subsets,
    build_estimator,
    evaluate_estimator,
    fmt_pct,
    json_safe,
    make_split,
    parse_folds,
    pct,
    write_summary,
)


def cache_key(paths, args):
    payload = {
        "paths": paths,
        "model_name": args.model_name,
        "sample_rate": args.sample_rate,
        "pooling": args.pooling,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def load_waveform_batch(paths, sample_rate, load_workers):
    if load_workers <= 1:
        return [load_audio_to_ram(path, sample_rate)[1][0].astype(np.float32) for path in paths]
    with ThreadPoolExecutor(max_workers=load_workers) as executor:
        loaded = list(executor.map(lambda p: load_audio_to_ram(p, sample_rate), paths))
    return [waveform[0].astype(np.float32) for _, waveform, _real in loaded]


def pool_ast_output(outputs, pooling):
    hidden = outputs.last_hidden_state.float()
    pooling = pooling.lower()
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        return hidden.mean(dim=1)
    if pooling == "cls_mean":
        return torch.cat([hidden[:, 0, :], hidden.mean(dim=1)], dim=1)
    raise ValueError(f"Unsupported pooling '{pooling}'. Use cls, mean, or cls_mean.")


def build_embedding_cache(selected_records, args, exp_dir):
    paths = sorted({r["path"] for r in selected_records})
    key = cache_key(paths, args)
    cache_path = exp_dir / f"ast_embedding_cache_{key}.npz"
    if cache_path.exists() and not args.rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = [str(p) for p in data["paths"]]
        if cached_paths == paths:
            print(f"[AST Cache] loaded {len(paths)} clips from {cache_path}")
            return {path: emb for path, emb in zip(cached_paths, data["embeddings"])}
        print("[AST Cache] cache path exists but path list changed; rebuilding.")

    cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else exp_dir / "hf_cache"
    if not cache_dir.is_absolute():
        cache_dir = (REPO_ROOT / cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[AST Setup] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print(
        "[AST Setup] "
        f"model={args.model_name} | device={device} | batch_size={args.embedding_batch_size} | "
        f"local_files_only={args.local_files_only}"
    )
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    embeddings = []
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, len(paths), args.embedding_batch_size):
            batch_paths = paths[start:start + args.embedding_batch_size]
            waveforms = load_waveform_batch(batch_paths, args.sample_rate, args.load_workers)
            inputs = extractor(
                waveforms,
                sampling_rate=args.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                outputs = model(**inputs)
                batch_embeddings = pool_ast_output(outputs, args.pooling)
            embeddings.append(batch_embeddings.float().cpu().numpy().astype(np.float32))
            done = min(start + args.embedding_batch_size, len(paths))
            if done == len(paths) or done % max(args.embedding_batch_size * 10, 1) == 0:
                elapsed = time.time() - start_time
                print(f"[AST Setup] embedded {done}/{len(paths)} clips in {elapsed:.1f}s")

    matrix = np.concatenate(embeddings, axis=0)
    np.savez_compressed(cache_path, paths=np.array(paths, dtype=object), embeddings=matrix)
    print(f"[AST Cache] saved {matrix.shape[0]}x{matrix.shape[1]} embeddings to {cache_path}")
    return {path: emb for path, emb in zip(paths, matrix)}


def main():
    parser = argparse.ArgumentParser(description="AST AudioSet-pretrained embedding probe for UrbanSound8K.")
    parser.add_argument("--data_dir", default=default_data_dir())
    parser.add_argument("--exp_name", default="ast_embedding_probe")
    parser.add_argument("--folds", default="1-3")
    parser.add_argument("--protocol", default="source_group_8_1_1")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--models", default="logreg,linear_svm,rbf_svm")
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--pooling", default="cls_mean", choices=["cls", "mean", "cls_mean"])
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--load_workers", type=int, default=4)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--max_test_clips", type=int, default=None)
    parser.add_argument("--min_source_support", type=int, default=5)
    parser.add_argument("--weak_group_limit", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = REPO_ROOT / "experiments" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(args.data_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(args.data_dir, "audio")
    clip_records = parse_dataset(csv_path, audio_base, CLASS_NAMES)
    folds = parse_folds(args.folds)
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]

    splits = []
    selected_records_by_path = {}
    for fold in folds:
        split = make_split(clip_records, fold, args.protocol, seed=args.seed)
        split = apply_smoke_subsets(split, args)
        splits.append(split)
        for record in split["train"] + split["val"] + split["test"]:
            selected_records_by_path[record["path"]] = record
        overlap = source_label_overlap_summary(split["train"], split["test"])
        val_overlap = source_label_overlap_summary(split["train"], split["val"]) if split["uses_validation"] else {"count": None}
        print(
            f"[Split] fold={fold} protocol={args.protocol} "
            f"clips train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} "
            f"train/test source-label overlap={overlap['count']} "
            f"train/val source-label overlap={val_overlap['count']}"
        )

    embedding_by_path = build_embedding_cache(list(selected_records_by_path.values()), args, exp_dir)
    feature_dim = len(next(iter(embedding_by_path.values())))

    fold_results = []
    for split in splits:
        print(f"\n=================== AST EMBEDDING PROBE FOLD {split['fold']} ===================")
        model_results = {}
        for name in model_names:
            estimator = build_estimator(name, args.seed + split["fold"], args.n_jobs, args.n_estimators)
            result = evaluate_estimator(name, estimator, split, embedding_by_path, args)
            model_results[name] = result
            print(
                f"{name:<12} | Val={fmt_pct(pct(result['val_acc']))} | "
                f"Test={fmt_pct(pct(result['test_acc']))} | "
                f"Worst={result['worst_class']} {fmt_pct(pct(result['worst_class_acc']))} | "
                f"Fit={result['fit_seconds']:.1f}s"
            )
        fold_results.append(
            {
                "fold": split["fold"],
                "protocol": args.protocol,
                "val_bucket": split["val_bucket"],
                "counts": {
                    "train": len(split["train"]),
                    "val": len(split["val"]),
                    "test": len(split["test"]),
                },
                "models": model_results,
            }
        )

    summary = {
        "exp_name": args.exp_name,
        "protocol": args.protocol,
        "seed": args.seed,
        "folds": folds,
        "models": model_names,
        "feature_dim": feature_dim,
        "feature_extractor": {
            "backend": "ast_embedding",
            "model_name": args.model_name,
            "sample_rate": args.sample_rate,
            "pooling": args.pooling,
            "amp": args.amp,
        },
        "fold_results": fold_results,
    }
    summary["aggregate"] = aggregate_results(fold_results, model_names)
    json_path, md_path = write_summary(summary, exp_dir)
    print(f"\nSummary written: {json_path}")
    print(f"Summary report : {md_path}")


if __name__ == "__main__":
    main()
