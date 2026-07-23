# Documentation hub

```text
docs/
  main/           ← canonical deploy path + promote-to-main
  data/           ← dataset profile + analysis standard (REQUIRED)
  experiments/    ← REGISTRY + historical notes
  architecture/
  hardware/
  reproduction/
  notebooks/
```

## Start here

| # | Doc |
|--:|---|
| 1 | [main/README.md](main/README.md) |
| 2 | [data/README.md](data/README.md) — **phân tích dữ liệu & kết quả chuẩn** |
| 3 | [main/SETUP_DEPLOY_SOURCE_SAFE.md](main/SETUP_DEPLOY_SOURCE_SAFE.md) |
| 4 | [main/SERVER_POLICY.md](main/SERVER_POLICY.md) — server `.9`, `main` only |
| 5 | [data/MULTIDATASET_PHASE1.md](data/MULTIDATASET_PHASE1.md) — ESC-50 + Speech Commands contract |
| 6 | [main/DECISIONS_LOG.md](main/DECISIONS_LOG.md) |
| 7 | [experiments/REGISTRY.md](experiments/REGISTRY.md) |
| 8 | [../configs/INDEX.md](../configs/INDEX.md) |
| 9 | [main/PROMOTE_TO_MAIN.md](main/PROMOTE_TO_MAIN.md) — historical, promotion completed |

## Configs

- **Run:** `configs/kv260_ds1d_pyramid_mixup_ema_val.json`
- **Catalog:** [configs/INDEX.md](../configs/INDEX.md)

## Experiments

- Local: `experiments/<exp_name>/`
- Metrics on git: `results/*` branches
- Always: REGISTRY before new run; ANALYSIS_STANDARD after run
