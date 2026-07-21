# Promote `research/fpga-1dcnn-90acc` → `main`

## Reality check

| Branch | Role |
|---|---|
| `research/fpga-1dcnn-90acc` | **Active development** (canonical docs, DS1D path) |
| `main` | Older TCAM/reproduce-oriented history; **diverged** from research |

As of last check: histories **diverged** (not a trivial fast-forward).  
Promotion must be **deliberate** (merge/PR), not casual `push --force` without review.

## What “promote to main” means here

Bring **main** up to the **canonical deploy path**:

- Main config + source-safe protocol docs  
- Experiment REGISTRY / config INDEX  
- Train entry + src needed for DS1D baseline  
- **Not** every rejected texture config noise (optional)  
- **Not** large `experiments/*.pt`  

Results metrics stay on `results/*` branches or docs tables — optional submodule later.

## Preconditions (analysis gate)

Before any merge to `main`:

- [ ] [../data/ANALYSIS_STANDARD.md](../data/ANALYSIS_STANDARD.md) applied on baseline H0  
- [ ] REGISTRY marks H0 as BASELINE; rejects documented  
- [ ] No secret/large files staged  
- [ ] `configs/kv260_ds1d_pyramid_mixup_ema_val.json` is the advertised main config  
- [ ] README canonical table matches docs/main  

## Recommended procedure (safe)

### Option A — Pull request (preferred)

```bash
# local
git fetch origin
git checkout research/fpga-1dcnn-90acc
git pull origin research/fpga-1dcnn-90acc

# open PR: research/fpga-1dcnn-90acc → main
# Title: Promote source-safe DS1D deploy path and doc registry to main
```

PR description checklist:

1. Summary of main path (model, protocol, metric)  
2. Link `docs/main/DECISIONS_LOG.md`  
3. Note diverged history / conflict risk  
4. Confirm analysis standard for baseline  

### Option B — Local merge then push main

```bash
git fetch origin
git checkout main
git pull origin main
git merge origin/research/fpga-1dcnn-90acc
# resolve conflicts favoring research canonical docs + DS1D configs
# run a smoke: python train.py --fold 1 --config configs/kv260_ds1d_pyramid_mixup_ema_val.json --epochs 1 --max_train_clips 32 --max_val_clips 16 --max_test_clips 16
git push origin main
```

**Do not** force-push `main` unless team agrees.

### Option C — Soft promote (if merge too painful)

Keep `main` as archive; set default branch on GitHub to `research/fpga-1dcnn-90acc`  
or create `main` ← orphan snapshot of research only (nuclear; document).

## What stays on research until stable

- WIP texture/KD configs  
- Untracked teacher scripts  
- Ongoing experiment notes  

## After promote

1. Tag release: e.g. `deploy-baseline-h0-docs`  
2. Server worktrees: track default branch  
3. New work: branch from updated `main` (`feature/...`)  

## Server after promote

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
git fetch origin
git checkout main   # or research if default not switched
git pull
test -f configs/kv260_ds1d_pyramid_mixup_ema_val.json && echo OK_MAIN_CONFIG
test -f docs/main/README.md && echo OK_DOCS
```
