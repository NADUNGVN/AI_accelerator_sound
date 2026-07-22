# Distill student from AST teacher **v3** (SDP fold1)

**Teacher T\*** (best-val test among 8000 runs):  
`server8000_ast_teacher_v3_stage2_sdp811_f1_20ep` → **89.77%** best-val test  
Path on SERVER-02:

```text
experiments/server8000_ast_teacher_v3_stage2_sdp811_f1_20ep/fold_1/checkpoints/best/
```

**Student:** DS-Conv2D-H1 Pyramid (deployable)  
**Protocol:** `source_group_8_1_1`, seed **83**, fold **1**  
**KD:** offline soft logits on **train only**; val/test label-only; student best by **val**

---

## Step 1 — Cache teacher logits (train split)

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sound_ast
cd $HOME/Dung_TDTU/AI_accelerator_sound

# confirm teacher best exists
ls experiments/server8000_ast_teacher_v3_stage2_sdp811_f1_20ep/fold_1/checkpoints/best/

python tools/cache_ast_teacher_logits.py \
  --data_dir data/UrbanSound8K \
  --fold 1 \
  --protocol source_group_8_1_1 \
  --seed 83 \
  --teacher_checkpoint_template \
    experiments/server8000_ast_teacher_v3_stage2_sdp811_f1_20ep/fold_{fold}/checkpoints/best \
  --output_template \
    experiments/ast_teacher_logits_v3_sdp811/fold_{fold}/teacher_logits.pt \
  --splits train \
  --batch_size 16 \
  --num_workers 8 \
  --device cuda \
  --overwrite
```

Expect log: `teacher_acc=...` on train, `clips=6996`.

---

## Step 2 — Train student + KD

```bash
python tools/run_multifold.py \
  --config configs/student_ds1d_ast_v3_kd_sdp811_val.json \
  --exp_name server8000_student_ast_v3_kd_sdp811_f1_50ep \
  --folds 1 \
  --epochs 50 \
  --analyze \
  --eval_modes
```

Or:

```bash
python train.py \
  --fold 1 \
  --config configs/student_ds1d_ast_v3_kd_sdp811_val.json \
  --exp_name server8000_student_ast_v3_kd_sdp811_f1_50ep
```

Primary metric: **`test_acc_best_val_model`** vs Model B **80.00%**.

---

## Step 3 — Push metrics (not checkpoints)

```bash
cd $HOME/Dung_TDTU/AI_accelerator_sound
export LD_LIBRARY_PATH=""

git add -f \
  experiments/server8000_student_ast_v3_kd_sdp811_f1_50ep/fold_1/metrics.json \
  experiments/server8000_student_ast_v3_kd_sdp811_f1_50ep/fold_1/summary.md \
  experiments/ast_teacher_logits_v3_sdp811/fold_1/teacher_logits_summary.md

# optional: do NOT push teacher_logits.pt if large
git commit -m "Add student AST-v3 KD fold1 metrics (SDP 8-1-1 seed 83)"
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git push origin research/ast-teacher-mvp-rtx8000
```

---

## Notes

- `build_model` never loads AST; student stays DS1D.  
- Cache keys: `fold{fold}/{slice_file_name}`.  
- Mixup batches skip KD when `apply_to_mixup=false` (default here).  
