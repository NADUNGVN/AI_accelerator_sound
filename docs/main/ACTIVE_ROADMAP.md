# Active roadmap (main path only)

## Done (do not redo blindly)

- [x] Lock model + source-safe baseline (local + server H0)
- [x] Texture T1/T2/H2 ablations → rejected vs H0
- [x] Demote official paper_9_1 full-10 from main path
- [x] Doc structure + experiment registry
- [x] Data analysis standard (`docs/data/`) + promote-to-main procedure

## Next (in order) — Phase A accuracy first

See [THREE_ACCURACY_TRACKS.md](THREE_ACCURACY_TRACKS.md).

1. **Track 1 — Single 80–85%**  
   Improve `test_acc_best_val_model` under source-safe + val (H0 server ~76.9%, local ~79.1%).  
   Gate: ≥80% fold1, then multi-fold; analysis per [../data/ANALYSIS_STANDARD.md](../data/ANALYSIS_STANDARD.md).

2. **Track 2 — Ensemble 80–85%**  
   Same runs report `test_acc_ensemble` (last-2). Local already ~79.9%; push server ensemble into band without faking single.

3. **Track 3 — Distill teacher → DS1D 80–85%**  
   Teacher already ~90%+ (AST train/cache). Reuse logits/ckpts; student best-val test target 80–85% (kdprotect f1 already ~80%).

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
