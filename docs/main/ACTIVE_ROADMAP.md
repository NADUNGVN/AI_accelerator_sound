# Active roadmap (main path only)

## Done (do not redo blindly)

- [x] Lock model + source-safe baseline (local + server H0)
- [x] Texture T1/T2/H2 ablations → rejected vs H0
- [x] Demote official paper_9_1 full-10 from main path
- [x] Doc structure + experiment registry
- [x] Data analysis standard (`docs/data/`) + promote-to-main procedure

## Next (in order) — Phase A accuracy first

See [THREE_ACCURACY_TRACKS.md](THREE_ACCURACY_TRACKS.md).

0. **Numbers in main** — [ACHIEVED.md](ACHIEVED.md): single **79.08%**, ens **79.89%**, KD student **80.00%** (one table, no local/server split).

1. **Track 1 — Single 80–85%**  
   From **79.08%** → ≥80% (then 85%). Same stack; analysis [../data/ANALYSIS_STANDARD.md](../data/ANALYSIS_STANDARD.md).

2. **Track 2 — Ensemble 80–85%**  
   From **79.89%** → ≥80% (then 85%). Same runs as Track 1 when possible.

3. **Track 3 — Distill teacher → DS1D 80–85%**  
   Already **80.00%** student; push toward **85%**. Teacher ~90%+ reuse only.

4. **Phase B (only after Phase A credible)**  
   SoC design, quantization, KV260 deploy, board latency/power.

5. **Promote git `main`** when Phase A path + docs stable — [PROMOTE_TO_MAIN.md](PROMOTE_TO_MAIN.md).

6. **Do not** full-10 paper_9_1 unless side-table needed.

## Command template (main)

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
git pull origin research/fpga-1dcnn-90acc
conda activate sound_env
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

python tools/run_multifold.py \
  --config configs/kv260_ds1d_pyramid_mixup_ema_val.json \
  --exp_name <NEW_NAME_NOT_REUSING_REJECTED> \
  --folds 1 \
  --epochs 50 \
  --analyze \
  --eval_modes
```

Before any new exp: check [../experiments/REGISTRY.md](../experiments/REGISTRY.md).
