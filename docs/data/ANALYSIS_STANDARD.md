# Standard analysis after every experiment

## 0. Preconditions

- [ ] `exp_name` not reusing a **REJECTED** recipe ([../experiments/REGISTRY.md](../experiments/REGISTRY.md))  
- [ ] Config listed in [../../configs/INDEX.md](../../configs/INDEX.md)  
- [ ] Same protocol as comparison baseline (usually `source_group_8_1_1`)  

## 1. Protocol & data integrity (`metrics.json`)

| Field | Expect (main path) |
|---|---|
| `protocol` | `source_group_8_1_1` |
| `uses_validation` | `true` |
| `source_label_overlap_train_test.count` | **0** |
| `model_name` | `kv260_audio_net_ds1d` (unless documented otherwise) |
| `distillation` | `null` for no-teacher; set for KD |
| `train_clip_count` / `val_clip_count` / `test_clip_count` | e.g. ~6996 / ~866 / ~870 on fold1 H0 |

## 2. Primary metrics

| Priority | Field | Main path use |
|---:|---|---|
| 1 | `test_acc_best_val_model` | **Gate vs H0 76.90%** |
| 2 | `best_val_clip_acc` + epoch of best val | Selection quality |
| 3 | `test_acc_last_snapshot` | Secondary |
| 4 | `test_acc_ensemble` | Secondary / research table |

**Decision gate (main):**

| Outcome | Action |
|---|---|
| best-val test ≥ H0 and worst-class not collapsed | Candidate; register BASELINE or IMPROVED |
| within −1 pp, confusion improves | Partial; limited follow-up |
| < H0 − 1 pp or worst-class crash | **REJECTED**; document lesson; do not stack |

## 3. Learning curves (`history.json`)

- Best val epoch (early vs late)  
- Train acc under mixup (often 60–70% — not raw fit)  
- Signs of collapse (val stuck ~10%)  

## 4. Errors (`analysis_all_cycles.json` / multifold per-class)

- Worst class + acc  
- Top confusions (e.g. AC→engine, street→children)  
- Compare to H0 confusions — did the intervention fix the **stated** failure mode?  

## 5. Commands (server or local)

```bash
EXP=experiments/<exp_name>
python - <<'PY'
import json
from pathlib import Path
import sys
exp = Path(sys.argv[1])
m = json.load(open(exp/"fold_1"/"metrics.json", encoding="utf-8-sig"))
print("protocol", m.get("protocol"), "uses_val", m.get("uses_validation"))
print("overlap", m.get("source_label_overlap_train_test"))
print("clips", m.get("train_clip_count"), m.get("val_clip_count"), m.get("test_clip_count"))
print("best_val", None if m.get("best_val_clip_acc") is None else round(m["best_val_clip_acc"]*100,2))
print("best_val_test", None if m.get("test_acc_best_val_model") is None else round(m["test_acc_best_val_model"]*100,2))
print("last", None if m.get("test_acc_last_snapshot") is None else round(m["test_acc_last_snapshot"]*100,2))
print("ens", None if m.get("test_acc_ensemble") is None else round(m["test_acc_ensemble"]*100,2))
print("model", m.get("model_name"), (m.get("model_params") or {}).get("params_with_bias"))
md = exp/"multifold_summary.md"
if md.exists():
    print(md.read_text(encoding="utf-8")[:2000])
PY
"$EXP"
```

Optional deep analysis (if not already from multifold `--analyze`):

```bash
python tools/analyze_experiment.py \
  --exp_dir experiments/<exp_name>/fold_1 \
  --fold 1 \
  --config configs/kv260_ds1d_pyramid_mixup_ema_val.json \
  --eval_all_cycles --eval_modes
```

## 6. Register & push

1. Fill [RESULT_REPORT_TEMPLATE.md](RESULT_REPORT_TEMPLATE.md) (short).  
2. Append row to [../experiments/REGISTRY.md](../experiments/REGISTRY.md).  
3. Push metrics:

```bash
# see docs/main — force-add metrics only on results/* branch
```

## 7. Promote readiness

Only if analysis standard passes **and** improve vs H0 (or intentional new baseline):  
→ [../main/PROMOTE_TO_MAIN.md](../main/PROMOTE_TO_MAIN.md)  
