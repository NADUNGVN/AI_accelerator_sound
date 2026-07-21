# Main path (canonical)

**Phase A (now):** hit **80–85%** on three accuracy tracks (single / ensemble / KD).  
**Phase B (later):** SoC design, quantization, KV260 deploy — **after** Phase A is credible.

Full write-up: [THREE_ACCURACY_TRACKS.md](THREE_ACCURACY_TRACKS.md).

| Track | Target | Metric | **Best achieved (research)** |
|---|---|---|---|
| **1. Single model** | **80–85%** | best-val → test | **79.08%** — [ACHIEVED.md](ACHIEVED.md) |
| **2. Ensemble** | **80–85%** | last-2 ensemble | **79.89%** |
| **3. Distill teacher → 1D-CNN** | **80–85% student** | student best-val test | **80.00%** (ens 80.23%); teacher ~90%+ |

Shared stack: `kv260_audio_net_ds1d` full-clip, source-safe + **val**, seed 83.

| Item | Canonical choice |
|---|---|
| Model | `kv260_audio_net_ds1d` full-clip (~101.7k params, ~61.9M MAC/clip) |
| Config | [`configs/kv260_ds1d_pyramid_mixup_ema_val.json`](../../configs/kv260_ds1d_pyramid_mixup_ema_val.json) (+ KD configs for Track 3) |
| Protocol | `source_group_8_1_1` + `fsid_classid_balanced_v1` |
| Seed | `83` (reproducibility only) |
| Splits | train / **val** / test |
| Checkpoint | **best validation** → report **test** of that checkpoint |
| Primary metric (Track 1) | `test_acc_best_val_model` |
| Ensemble metric (Track 2) | `test_acc_ensemble` |
| Teacher (Track 3) | train-time only; **deploy = student only** |

## Read next

1. [ACHIEVED.md](ACHIEVED.md) — **what is already achieved (single table)**  
2. [THREE_ACCURACY_TRACKS.md](THREE_ACCURACY_TRACKS.md) — **three goals 80–85%**  
3. [SETUP_DEPLOY_SOURCE_SAFE.md](SETUP_DEPLOY_SOURCE_SAFE.md) — full setup lock  
3. [../data/README.md](../data/README.md) — **data + result analysis standard**  
4. [DECISIONS_LOG.md](DECISIONS_LOG.md) — what we accepted / rejected  
5. [ACTIVE_ROADMAP.md](ACTIVE_ROADMAP.md) — what to run next  
6. [../experiments/REGISTRY.md](../experiments/REGISTRY.md) — all exp names + status  
7. [../../configs/INDEX.md](../../configs/INDEX.md) — config catalog by status  
8. [PROMOTE_TO_MAIN.md](PROMOTE_TO_MAIN.md) — merge research → git `main`  

## Not main path

- Official `paper_9_1` full-10 without val → optional literature only  
- Texture T1/T2/H2 recipes that lost to H0 → see REGISTRY **REJECTED**  
