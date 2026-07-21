# AI accelerator sound — UrbanSound8K 1D-CNN → KV260

Source-safe environmental sound classification with a **deployable 1D-CNN student** (~102k params), targeting **KV260**-class budgets.

**Paper hardware wording:** report **RTX 3090** for training compute used in this repo’s main results. Other GPUs (local / second 3090 / RTX 8000) are for lab throughput only and need not appear in the paper.

---

## Canonical path (main)

| Item | Value |
|------|--------|
| **Model** | `kv260_audio_net_ds1d` full-clip |
| **Params / MAC** | ~101.7k / ~61.9M |
| **Config** | `configs/kv260_ds1d_pyramid_mixup_ema_val.json` |
| **Protocol** | `source_group_8_1_1` + **val** + **best-val checkpoint** |
| **Seed** | 83 (reproducibility) |
| **Docs** | [`docs/main/README.md`](docs/main/README.md) |
| **Achieved numbers** | [`docs/main/ACHIEVED.md`](docs/main/ACHIEVED.md) |
| **Config catalog** | [`configs/INDEX.md`](configs/INDEX.md) |
| **Exp registry** | [`docs/experiments/REGISTRY.md`](docs/experiments/REGISTRY.md) |
| **Data analysis standard** | [`docs/data/README.md`](docs/data/README.md) |

### Phase A — accuracy targets (before SoC / quant / board)

| Track | Target | Best achieved (research) |
|-------|--------|--------------------------:|
| 1. Single model | 80–85% | **79.08%** best-val test |
| 2. Ensemble (last-2) | 80–85% | **79.89%** |
| 3. KD teacher → student | 80–85% student | **80.00%** (ens 80.23%); teacher ~90%+ |

### Phase B — later

SoC design → quantization → **KV260 deploy**.

---

## Repo layout

```text
train.py           # training entry
configs/           # MAIN config + INDEX (status tags); legacy configs kept for history
src/               # data, models, training, …
tools/             # multifold, analyze, FLOPs, …
scripts/           # server runners
docs/main/         # canonical decisions + ACHIEVED
docs/data/         # dataset + analysis checklist
docs/experiments/  # REGISTRY + notes
docs/architecture|hardware|reproduction|notebooks/
```

`experiments/`, `logs/`, `checkpoints/`, `data/` are **runtime artifacts** (gitignore). Do not commit large `.pt` files. Push light metrics on `results/*` branches with `git add -f` when needed.

---

## Quick start (source-safe fold 1)

```bash
pip install -r requirements.txt
# place UrbanSound8K under data/UrbanSound8K (see train defaults / server tar)

python tools/run_multifold.py \
  --config configs/kv260_ds1d_pyramid_mixup_ema_val.json \
  --exp_name my_run_f1_50ep \
  --folds 1 \
  --epochs 50 \
  --analyze \
  --eval_modes
```

Primary metric: **`test_acc_best_val_model`**. Secondary: last snapshot, ensemble.

---

## Not the main path

- Official `paper_9_1` full-10 without val (optional literature only)
- Rejected texture recipes (see REGISTRY)
- Random-split “high accuracy” without source-safe claim
- Old TCAM-only reproduce narrative as the project headline

Legacy reproduce configs remain under `configs/reproduce_*.json` for archaeology, not as the default entry.
