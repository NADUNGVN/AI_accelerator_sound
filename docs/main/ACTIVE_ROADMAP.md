# Active roadmap (main only)

## Operating lock

- Git branch for new work: `main`
- Default train/eval server: `CPU-FPGA-GPU` RTX 3090
- Do not dispatch to `.13`, RTX 8000, or KV260 unless explicitly assigned
- Server details: [SERVER_POLICY.md](SERVER_POLICY.md)

## Done (do not redo blindly)

- [x] Lock model + source-safe baseline (local + server H0)
- [x] Texture T1/T2/H2 ablations → rejected vs H0
- [x] Demote official paper_9_1 full-10 from main path
- [x] Doc structure + experiment registry
- [x] Data analysis standard (`docs/data/`) + promote-to-main procedure

## Next (in order)

See [THREE_ACCURACY_TRACKS.md](THREE_ACCURACY_TRACKS.md).

0. **Numbers in main** — [ACHIEVED.md](ACHIEVED.md): single **79.08%**, ens **79.89%**, KD student **80.00%** (one table, no local/server split).

1. **Phase 1 multi-dataset expansion**
   Initial loaders/configs exist for **ESC-50** and **Speech Commands**. Next gate: dataset availability + smoke split fingerprints on `CPU-FPGA-GPU` before any accuracy claim. Contract: [../data/MULTIDATASET_PHASE1.md](../data/MULTIDATASET_PHASE1.md).

2. **Track 1 — Single 80–85%**
   From **79.08%** → ≥80% (then 85%). Same stack; analysis [../data/ANALYSIS_STANDARD.md](../data/ANALYSIS_STANDARD.md).

3. **Track 2 — Ensemble 80–85%**
   From **79.89%** → ≥80% (then 85%). Same runs as Track 1 when possible.

4. **Track 3 — Distill teacher → DS1D 80–85%**
   Already **80.00%** student; push toward **85%**. Teacher ~90%+ reuse only.

5. **Phase B (only after Phase A credible)**
   SoC design, quantization, KV260 deploy, board latency/power.

6. **Promote git `main`** — completed; [PROMOTE_TO_MAIN.md](PROMOTE_TO_MAIN.md) is historical only.

7. **Do not** full-10 paper_9_1 unless side-table needed.

## Command template (main)

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
git fetch origin
git checkout main
git pull origin main
conda activate sound_env
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

python tools/run_multifold.py \
  --config configs/main/student_ds_conv2d_h1_pyramid_sourcegroup.json \
  --exp_name <NEW_NAME_NOT_REUSING_REJECTED> \
  --folds 1 \
  --epochs 50 \
  --analyze \
  --eval_modes
```

Before any new exp: check [../experiments/REGISTRY.md](../experiments/REGISTRY.md).
