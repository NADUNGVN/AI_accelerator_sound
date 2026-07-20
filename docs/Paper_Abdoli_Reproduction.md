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

## Commands

```bash
# Paper best (89% reported)
python train.py --fold 1 --config configs/paper_abdoli_gamma.json --exp_name paper_abdoli_gamma

# Paper random-init best (87% reported)
python train.py --fold 1 --config configs/paper_abdoli_rand.json --exp_name paper_abdoli_rand
```

Primary metric vs paper: `test_acc_best_val_model` in `experiments/.../metrics.json`.

## Known non-bit-identical differences

1. **Framework**: paper used TensorFlow/Keras; this repo uses PyTorch. Numerics (BN eps/momentum, Adadelta internals) can differ slightly.
2. **Gammatone toolbox**: paper used Ellis Gammatone-like spectrograms toolbox; we use a standard analytic ERB gammatone impulse response (same f-range, 64 bands, length 512).
3. **Weight init** (non-Gammatone layers): paper does not specify; we use Xavier uniform.
4. **Validation fold choice**: paper says “one of the nine training folds”; we use `val = (test_fold % 10) + 1`.
5. **Clip padding**: all clips are padded/truncated to 4 s @ 16 kHz before framing (UrbanSound8K max length is 4 s).

These should not change the architecture contract; expect fold-level variance around the published mean.
