# Main path (canonical)

**Goal:** deployable **1D-CNN student** on KV260-class budget, source-safe training, clear checkpoint selection.

| Item | Canonical choice |
|---|---|
| Model | `kv260_audio_net_ds1d` full-clip (~101.7k params, ~61.9M MAC/clip) |
| Config | [`configs/kv260_ds1d_pyramid_mixup_ema_val.json`](../../configs/kv260_ds1d_pyramid_mixup_ema_val.json) |
| Protocol | `source_group_8_1_1` + `fsid_classid_balanced_v1` |
| Seed | `83` (reproducibility only) |
| Splits | train / **val** / test |
| Checkpoint | **best validation** → report **test** of that checkpoint |
| Primary metric | `test_acc_best_val_model` |
| Secondary | last snapshot, last-2 ensemble |
| Teacher | optional later; **deploy = student only** |

## Read next

1. [SETUP_DEPLOY_SOURCE_SAFE.md](SETUP_DEPLOY_SOURCE_SAFE.md) — full setup lock  
2. [../data/README.md](../data/README.md) — **data + result analysis standard**  
3. [DECISIONS_LOG.md](DECISIONS_LOG.md) — what we accepted / rejected  
4. [ACTIVE_ROADMAP.md](ACTIVE_ROADMAP.md) — what to run next  
5. [../experiments/REGISTRY.md](../experiments/REGISTRY.md) — all exp names + status (avoid re-runs)  
6. [../../configs/INDEX.md](../../configs/INDEX.md) — config catalog by status  
7. [PROMOTE_TO_MAIN.md](PROMOTE_TO_MAIN.md) — merge research → git `main`  

## Not main path

- Official `paper_9_1` full-10 without val → optional literature only  
- Texture T1/T2/H2 recipes that lost to H0 → see REGISTRY **REJECTED**  
