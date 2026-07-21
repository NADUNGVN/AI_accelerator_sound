# Active roadmap (main path only)

## Done (do not redo blindly)

- [x] Lock model + source-safe baseline (local + server H0)
- [x] Texture T1/T2/H2 ablations → rejected vs H0
- [x] Demote official paper_9_1 full-10 from main path
- [x] Doc structure + experiment registry

## Next (in order)

1. **Stand on H0** — `experiments/server3090_notacher_f1_fullclip_baseline_50ep` (server results branch) as deploy baseline numbers.
2. **Only new runs** that can beat **76.90% best-val test** under same protocol/config family (or documented KD).
3. **Optional KD (Track T)** — reuse AST/logits; student still DS1D; report best-val test.
4. **Hardware metrics** — params/MAC already; latency/power when board ready.
5. **Do not** full-10 paper_9_1 unless side-table needed.

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
