# Server policy (main only)

**Status:** canonical for new runs after `main` promotion.

**Rule:** new sound training/evaluation starts from git `main`; do not dispatch work from research/reproduce branches.

## Active machine

| Role | Machine | Use |
|---|---|---|
| Default train/eval server | `CPU-FPGA-GPU` / RTX 3090 24 GB / i7-8700K 6c/12t / ~32 GB RAM | Main Phase A and Phase 1 dataset runs |
| Local Windows | developer smoke only | Tiny smoke runs, docs, code edits |
| 3090 `.13` | standby | Use only when explicitly assigned |
| RTX 8000 / `SERVER-02` | standby | Use only when explicitly assigned |
| KV260 | Phase B | Quantization/deploy after research metrics are credible |

## Server entry sequence

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
hostname   # must print CPU-FPGA-GPU for the default server
conda activate sound_env
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

git fetch origin
git checkout main
git pull origin main

test -f configs/main/student_ds_conv2d_h1_pyramid_sourcegroup.json && echo OK_MAIN_CONFIG
test -f docs/main/ACHIEVED.md && echo OK_MAIN_DOCS
```

Prefer `screen` for runs expected to last beyond an interactive SSH session:

```bash
screen -S sound_phase1
# run command
# detach: Ctrl-a d
# reattach: screen -r sound_phase1
```

## Loader/runtime defaults

For `CPU-FPGA-GPU`, the canonical main config uses:

```text
batch_size = 64
num_workers = 6
amp = true
```

`num_workers=0` is only for conservative/debug configs. If a server run stalls the GPU with low utilization, first check the effective config and DataLoader worker count before changing the model.

Local Windows smoke runs can use smaller settings, typically `batch_size=16..32` and `num_workers=2..4` if process spawning is stable.

## Artifact policy

Commit/push lightweight evidence only:

- `metrics.json`
- `history.json`
- `predictions.json`
- `analysis_all_cycles.json`
- `multifold_summary.json`
- `multifold_summary.md`
- short analysis notes under `docs/experiments/`

Do not add full checkpoints from `experiments/**/*.pt` unless a deliverable package is explicitly being prepared under `deploy/student_models/`.

## Multi-machine policy

Do not split Phase 1 jobs across `.13`, RTX 8000, or KV260 unless the owner explicitly asks for multi-machine scheduling. When sharing work later, each machine must have a unique experiment name and a written dataset/split contract before training starts.
