# Repository structure (inheritance-friendly)

This layout separates **stable public surfaces** (models, configs, paper docs) from **runtime artifacts** (experiments, checkpoints, logs) so new experiments and co-authors can extend the project without inventing parallel trees.

```text
1_ai_accelerator_sound/
├── README.md                 # Academic overview (paper premise)
├── STRUCTURE.md              # This file — how to extend the repo
├── requirements.txt
├── train.py                  # Single training entrypoint
│
├── configs/
│   ├── INDEX.md              # Status tags for all JSON configs
│   ├── main/                 # Canonical paper configs (preferred entry)
│   │   ├── student_ds_conv2d_h1_pyramid_sourcegroup.json
│   │   ├── student_ds_conv2d_h1_pyramid_clean811.json
│   │   ├── student_ds_conv2d_h1_pyramid_clean811_mcisr.json
│   │   └── student_ds_res1d_se_fullclip.json
│   └── *.json                # Historical / ablation configs (kept)
│
├── src/                      # Installable-style Python package
│   ├── data/                 # Dataset parse, frames, features
│   ├── models/               # Layer-named networks (+ legacy shims)
│   │   ├── ds_conv2d_h1_pyramid.py   # MAIN student
│   │   ├── ds_res1d_se.py            # pure Conv1d baseline
│   │   ├── ast_transformer_teacher.py
│   │   ├── tcam_attn1d.py            # literature baseline
│   │   ├── kv260_ds1d.py             # shim → ds_conv2d_h1_pyramid
│   │   ├── efficient1dcnn.py         # shim → ds_res1d_se
│   │   └── tcam1dcnn.py              # shim → tcam_attn1d
│   ├── training/             # Trainer, losses (incl. MC-ISR)
│   ├── evaluation/           # Eval helpers (extend here)
│   └── utils/
│
├── tools/                    # CLI utilities (multifold, analyze, figures)
├── scripts/                  # Server / multi-run shell & PowerShell
│
├── docs/
│   ├── paper/                # Paper-facing cards + figures
│   │   ├── DATA.md
│   │   ├── MODELS.md
│   │   └── figures/*.svg
│   ├── main/                 # Canonical decisions, ACHIEVED, tracks
│   ├── data/                 # Dataset + analysis standard
│   ├── architecture/         # FLOPs methodology, design notes
│   ├── experiments/          # REGISTRY + run write-ups
│   ├── hardware/             # Board / server notes
│   ├── reproduction/         # Leakage / audit trail
│   └── notebooks/
│
├── results/                  # Light, shareable metrics & figures
│   ├── figures/
│   ├── metrics/
│   └── …
│
├── experiments/              # Runtime run roots (gitignored .pt)
├── checkpoints/              # Ad-hoc ckpts (gitignored)
├── logs/                     # Train logs (gitignored bulk)
└── data/                     # Datasets (not in git)
```

---

## How to inherit / extend cleanly

### Add a new model

1. Create `src/models/<layer_family>.py` with a **layer-descriptive** class name.  
2. Export it from `src/models/__init__.py`.  
3. Register paper + optional legacy keys in `train.py` → `build_model`.  
4. Add a config under `configs/main/` if it becomes a paper baseline.  
5. Document operators + params/MACs in `docs/paper/MODELS.md`.  
6. Keep old names only as **aliases**, never delete metrics history.

### Add a new experiment

1. Copy a `configs/main/*.json` config; change only intentional fields.  
2. Run via `tools/run_multifold.py` with a unique `--exp_name`.  
3. Register the name + status in `docs/experiments/REGISTRY.md`.  
4. Analyze with `tools/analyze_experiment.py` using `docs/data/ANALYSIS_STANDARD.md`.  
5. If the run becomes a claim, update `docs/main/ACHIEVED.md` and regenerate figures:

```bash
python tools/generate_paper_figures.py
```

### Add a new dataset

1. Implement loader under `src/data/` (do not hard-code only US8K paths in models).  
2. Document seed/split status in `docs/paper/DATA.md` (**mark NOT SEEDED until trained**).  
3. Add config + registry entry; keep US8K MAIN configs untouched.

### What not to commit

- Large `.pt` checkpoints  
- Full `experiments/**` audio caches  
- Local absolute paths in committed configs  

Push light metrics with `git add -f` only when needed for evidence branches.

---

## Canonical entrypoints

| Task | Command / path |
|---|---|
| Train MAIN student (source-group) | `tools/run_multifold.py --config configs/main/student_ds_conv2d_h1_pyramid_sourcegroup.json` |
| Train clean811 | `configs/main/student_ds_conv2d_h1_pyramid_clean811.json` |
| Train MC-ISR | `configs/main/student_ds_conv2d_h1_pyramid_clean811_mcisr.json` |
| Achieved numbers | `docs/main/ACHIEVED.md` |
| Paper figures | `docs/paper/figures/` |
| Complexity methodology | `docs/paper/MODELS.md` §4 |
