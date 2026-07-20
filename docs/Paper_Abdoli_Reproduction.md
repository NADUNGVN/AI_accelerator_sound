# Paper-Faithful Reproduction — Abdoli et al. 2019

Source: arXiv:1904.08990v1, *End-to-End Environmental Sound Classification using a 1D Convolutional Neural Network*.

## What was aligned

| Paper detail | Implementation |
|---|---|
| Table 1 topology (16k / 16kG) | `src/models/abdoli1dcnn.py` |
| BN **after** ReLU on each conv | `Conv → ReLU → BN → (Pool)` |
| FC 128→64→10, dropout 0.25 | same |
| Gammatone CL1: 64×512, 100 Hz–8 kHz | `src/utils/gammatone.py` |
| CL1 non-trainable (best setup) | `freeze_gammatone=true` freezes weight **and** bias |
| Params rand = 256,538 | verified |
| Params gamma (frozen) = 550,506 | verified |
| MSLE loss Eq. 4 | `MSLELoss` in `train.py` |
| Adadelta lr=1.0, batch 100, ≤100 epochs | paper configs |
| Early stopping on validation | `patience` in config |
| 50% overlap (gamma best) | `frame_hop=8000` |
| 75% overlap (rand best 87%) | `frame_hop=4000` |
| Rectangular window | no taper applied (identity = rectangular) |
| Sum-rule frame aggregation | `Trainer.evaluate_clips(aggregation="sum")` |
| 10-fold CV, 1 fold as val | protocol `clean_8_1_1` |
| No snapshot ensemble | `cycles=1` for paper configs |

## Framing (no TCAM 4 s zero-canvas)

| Setting | Paper Abdoli config |
|---|---|
| `pad_to_seconds` | `null` — keep **real** clip length |
| `max_seconds` | `4.0` — truncate only if longer than US8K max |
| `skip_near_zero_frames` | `true` |
| `zero_abs_threshold` | `1e-4` (peak \|x\|) |
| Short clip `< L` | one frame, right-pad that frame only |
| All candidates silent | keep highest-energy frame (never drop clip) |
| Hard checks | gamma/rand require `L=16000` and paper hop unless `allow_non_paper_framing` |

Legacy TCAM runs (non-Abdoli `model_name`) still default to `pad_to_seconds=4.0`.

Train log prints: duration stats, kept/candidate frames, near-zero skip rate.

## Commands

```bash
# Paper best (89% reported)
python train.py --fold 1 --config configs/paper_abdoli_gamma.json --exp_name paper_abdoli_gamma

# Paper random-init best (87% reported)
python train.py --fold 1 --config configs/paper_abdoli_rand.json --exp_name paper_abdoli_rand
```

Primary metric vs paper: `test_acc_best_val_model` in `experiments/.../metrics.json`.

## Known non-bit-identical differences

1. **Framework**: paper used TensorFlow/Keras; this repo uses PyTorch.
2. **Gammatone toolbox**: paper used Ellis toolbox; we use analytic ERB gammatone (100 Hz–8 kHz, 64×512).
3. **Weight init** (non-Gammatone layers): Xavier uniform (unspecified in paper).
4. **Validation fold**: `val = (test_fold % 10) + 1`.
5. Near-silent skip threshold is an engineering choice not stated in the paper (avoids training pure-zero frames as class labels).

Expect fold-level variance around the published mean.
