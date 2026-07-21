# Decisions log (locked)

## Accept

| Decision | Evidence |
|---|---|
| Main model = `kv260_audio_net_ds1d` full-clip ~102k | Beats frame16k, MIL no-teacher; within KV260 budget |
| Main protocol = `source_group_8_1_1` + val + best-val test | Deploy-oriented; source leak = 0; checkpoint selectable |
| Keep `seed=83` | Reproducible splits/init — not “official paper seed work” |
| **Three accuracy tracks @ 80–85% before SoC/quant/KV260** | See [THREE_ACCURACY_TRACKS.md](THREE_ACCURACY_TRACKS.md) |
| Track 1: **single** model best-val test 80–85% | Current ~77–79% fold1 |
| Track 2: **ensemble** (last-2) 80–85% | Local ~79.9% near band |
| Track 3: **KD** teacher→student 80–85% (teacher already ~90%+) | kdprotect f1 ~80%; teacher AST ~92% train/cache |
| Deploy story = **one** student checkpoint | Ensemble/KD teacher are train-time or multi-ckpt research |
| Official `paper_9_1` without val | **Not required** for hardware main path; optional side table |
| SoC / quantization / KV260 | **Phase B** after accuracy tracks |

## Reject / do not re-run as main

| ID | Exp / config | Why |
|---|---|---|
| T1 | `server3090_texture_sourcebalance_f1_50ep` / `..._sourcebalance_ce_val.json` | best-val test 73.56% < H0 76.90% |
| T2 | `server3090_texture_sourcehard_f1_50ep` / `..._sourcehard_ce_val.json` | Confounded multi-HP; jackhammer collapse |
| H2 | `server3090_texture_h2_hneg_ac_engine_f1_50ep` / `..._hneg_ac_engine_val.json` | best-val test 74.14%; AC→engine worse |
| H3/H4/LR texture | configs prepared | Do not run unless new hypothesis; H2 already rejected axis |
| paper91 full-10 | `..._mixup_ema_paper91.json` smoke fold1 ~67% last | Not main; optional side table only |
| Random clip protocol | high acc | Leakage; not thesis headline |
| MIL no-teacher as replacement | ~71% ens local | Under full-clip DS1D |

## Hold / optional later

| Item | Note |
|---|---|
| AST teacher + logits f1–f2 | Reuse; do not retrain if files exist |
| KD student (kdprotect ~80% f1 local) | Track T later under same source-safe + val |
| Official 10-fold mean | Only if reviewer demands literature table |

## Metric rules

| Path | Primary metric |
|---|---|
| Main (source-safe) | `test_acc_best_val_model` |
| Optional paper_9_1 | `test_acc_last_snapshot` (+ ensemble secondary) |
| Never | Select checkpoint using test-set accuracy during training |
