# Documentation hub

```text
docs/
  main/                 ← START HERE (deploy / thesis canonical path)
  experiments/
    REGISTRY.md         ← all exp names + BASELINE / REJECTED / OPTIONAL
    *.md                ← detailed notes (historical + active)
  architecture/         ← model / FLOPs / KV260 design
  hardware/             ← server / board notes
  reproduction/         ← paper reproduce & leakage analysis
  notebooks/            ← dataflow research notebook
```

## Start here

| Order | Doc |
|---:|---|
| 1 | [main/README.md](main/README.md) |
| 2 | [main/SETUP_DEPLOY_SOURCE_SAFE.md](main/SETUP_DEPLOY_SOURCE_SAFE.md) |
| 3 | [main/DECISIONS_LOG.md](main/DECISIONS_LOG.md) |
| 4 | [main/ACTIVE_ROADMAP.md](main/ACTIVE_ROADMAP.md) |
| 5 | [experiments/REGISTRY.md](experiments/REGISTRY.md) |
| 6 | [../configs/INDEX.md](../configs/INDEX.md) |

## Configs

- **Run this:** `configs/kv260_ds1d_pyramid_mixup_ema_val.json`
- **Catalog:** [configs/INDEX.md](../configs/INDEX.md)  
- Paths of old configs **unchanged** so past commands still resolve; status is in INDEX.

## Experiments on disk

- Local: `experiments/<exp_name>/` (gitignored)
- Pushed metrics: git branches `results/*`
- Always check REGISTRY before a new `exp_name`

## Design principle

| Layer | Stable path? | How we avoid chaos |
|---|---|---|
| Code `src/`, `train.py` | yes | single entry |
| Configs | yes (filenames fixed) | INDEX status tags |
| Docs | hub + main/ | decisions in one place |
| Exp outputs | name-based | REGISTRY status |
