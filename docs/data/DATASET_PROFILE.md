# Dataset profile — UrbanSound8K (project lock)

## Scale

| Item | Value |
|---|---|
| Clips | **8732** |
| Classes | **10** (air_conditioner … street_music) |
| Official folds | **10** predefined folders |
| Metadata | `UrbanSound8K.csv` (`fsID`, `fold`, `classID`, salience, …) |
| Source of excerpts | Freesound recordings (`fsID`) — many clips share one source |

## Why structure matters

1. **Same `fsID` in train and test** → model can learn recording fingerprint (leak).  
2. Official folds reduce leak vs pure random shuffle, but **are not identical** to our source-group split.  
3. Class balance and duration vary → need class weights / careful metrics (worst class).

## Audio pipeline (main model)

| Step | Setting |
|---|---|
| Resample | 16 kHz mono |
| Length | 4 s → **64000** samples |
| Framing (main) | **1 frame/clip** (full-clip) |
| Label | clip-level 10-way |

## Project data locations

| Env | Typical path |
|---|---|
| Server 3090 | `data/UrbanSound8K` (from `UrbanSound8K_on_server.tar.gz`) |
| Workspace guides | `1_ai_accelerator_sound/data/UrbanSound8K` |

## Prior analyses (link, do not redo from zero)

| Topic | Where |
|---|---|
| Split effect / leakage | workspace `docs_workspace/30_dataflow_split/SPLIT_EFFECT_*` + `docs/reproduction/UrbanSound8K_Leakage_*` |
| Silent / padding frames | workspace `SILENT_FRAME_*` under `docs_workspace/30_dataflow_split/` |
| Dataflow figures | `docs/notebooks/UrbanSound8K_1D_CNN_Dataflow_Research.ipynb` |
| Source-group audit | `docs/experiments/KV260_DS1D_Source_Group_Audit.md` |

## Minimum verification before train

```bash
# counts
find data/UrbanSound8K/audio -name '*.wav' | wc -l   # expect 8732
test -f data/UrbanSound8K/metadata/UrbanSound8K.csv && echo OK_META
```

After one training fold, in `metrics.json` check:

- `train_clip_count`, `val_clip_count`, `test_clip_count` > 0 (val=0 only if paper_9_1 optional)  
- `source_label_overlap_train_test.count == 0` for source-group protocols  
