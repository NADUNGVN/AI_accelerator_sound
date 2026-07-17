# KV260 1D-CNN Architecture Design

## Scope

There are two active directions:

1. **Current efficient full-clip model**: already implemented as `EfficientAudioCNN1D`.
2. **Next KV260-safe design**: still 1D-CNN logically, but represented as Conv2D with height 1 for Vitis AI / DPU deployment.

Both directions use the same classification task:

```text
Input audio: mono waveform
Sample rate: 16 kHz
Clip length: 4 seconds
Input samples: 64,000
Classes: 10 UrbanSound8K classes
Output: 10 logits, softmax applied only for evaluation
```

## Direction 1: Current Efficient Full-Clip 1D-CNN

### Why This Direction Exists

The original TCAM paper-style model uses 15 frames per clip. In our implementation:

```text
TCAM cost = about 230.19M MAC/frame x 15 frames = about 3.45B MAC/clip
```

That is too expensive for the final KV260 goal, even if parameters are only about 409K.

The current efficient model processes the full 4-second waveform in one pass:

```text
Efficient full-clip cost = about 98.66M MAC/clip
Params = 149,088
```

This meets the initial KV260 Excellent-tier budget: `<=100M MAC/clip` and `<=150K params`.

### Input

```text
Raw waveform tensor: [B, 1, 64000]
B = batch size
1 = mono channel
64000 = 4 seconds x 16000 Hz
```

No spectrogram, MFCC, mel filterbank, or handcrafted feature extraction is used. This keeps the model in the 1D-CNN family.

### Layer Flow

| Stage | Operation | Output shape | MACs | Why |
|---|---|---:|---:|---|
| Input | Raw waveform | `[B, 1, 64000]` | 0 | Direct 1D audio input |
| Stem branch 1 | Conv1D k=9, s=4, 8 ch | `[B, 8, 16000]` | 1.15M | short transient patterns |
| Stem branch 2 | Conv1D k=31, s=4, 8 ch | `[B, 8, 16000]` | 3.97M | medium waveform patterns |
| Stem branch 3 | Conv1D k=63, s=4, 8 ch | `[B, 8, 16000]` | 8.06M | wider low-level temporal context |
| Stem concat | Concatenate channels | `[B, 24, 16000]` | 0 | multi-scale low-level features |
| Block 1 | DSRes k=15, s=2, 24->32 | `[B, 32, 8000]` | 15.17M | early downsample, preserve transients |
| Block 2 | DSRes k=15, s=2, 32->48 | `[B, 48, 4000]` | 14.21M | larger feature bank |
| Block 3 | DSRes k=11, s=2, 48->64 | `[B, 64, 2000]` | 13.35M | mid-level sound events |
| Block 4 | DSRes k=9, s=2, 64->96 | `[B, 96, 1000]` | 12.87M | compress time, expand channels |
| Block 5 | DSRes k=9, s=2, 96->128 | `[B, 128, 500]` | 12.72M | higher-level event patterns |
| Block 6 | DSRes k=7, s=2, 128->160 | `[B, 160, 250]` | 10.47M | compact event sequence |
| Block 7 | DSRes k=7, s=1, 160->160 | `[B, 160, 250]` | 6.69M | refine high-level features |
| Pool | mean + max over time | `[B, 320]` | small | summarize full 4-second clip |
| Classifier | Linear 320->10 | `[B, 10]` | 0.003M | class logits |

Total:

```text
Params: 149,088
MACs/clip: 98,656,256
```

### What A DSRes Block Does

Each block is:

```text
Depthwise Conv1D -> BatchNorm -> SiLU
Pointwise Conv1D -> BatchNorm -> SiLU
Squeeze-Excitation
Dropout1D
Residual add
SiLU
```

Meaning:

- **Depthwise convolution** learns temporal filters per channel at low cost.
- **Pointwise convolution** mixes information across channels.
- **Squeeze-Excitation** reweights useful channels.
- **Residual path** makes deeper training more stable.
- **Dropout** reduces overfit on source-independent split.

### Weakness For KV260 Deployment

This model is good for research and quick accuracy/compute tradeoff, but it is not yet the final deployable architecture because:

- it uses PyTorch `Conv1d`;
- it uses SiLU;
- it uses Squeeze-Excitation with sigmoid;
- it uses mean+max pooling;
- Vitis AI might partition unsupported operators to CPU.

Therefore, Direction 1 is the **research efficient baseline**, not yet the final KV260-safe graph.

## Direction 2: Next KV260-Safe 1D-CNN

### Core Design Rule

Keep the model logically 1D-CNN, but implement/export it as:

```text
Input: [B, C, 1, T]
Conv2D kernel: [1, k]
Stride: [1, s]
```

This is still convolution along time only. The height dimension is fixed to 1.

Why:

- DPU toolchains are CNN/Conv2D-oriented.
- Conv2D-H1 is more likely to compile cleanly than arbitrary Conv1D graphs.
- It preserves the 1D waveform-processing claim.

### Deployment-Safe Block

Preferred block:

```text
Depthwise Conv2D-H1 -> BatchNorm -> ReLU
Pointwise Conv2D 1x1 -> BatchNorm -> ReLU
```

Avoid in the first deployable version:

- SiLU/Swish;
- sigmoid channel attention;
- complex pooling;
- unnecessary residual additions;
- custom operators.

Residuals and attention can be added only after Vitis AI Model Inspector confirms that the graph stays on DPU.

### Proposed Architecture: KV260AudioNet-DS1D

Input:

```text
[B, 1, 1, 64000]
```

Layer table:

| Stage | Operation | Input -> Output | Receptive field | MACs | Why |
|---|---|---:|---:|---:|---|
| Input | waveform | `[B,1,1,64000]` | 1 sample | 0 | raw 4-second audio |
| Stem | Conv2D-H1 k=31, s=4, 1->24 | `[B,1,1,64000] -> [B,24,1,16000]` | 31 | 11.90M | first waveform filters, early downsample |
| B1.DW | depthwise k=15, s=2 | `[B,24,1,16000] -> [B,24,1,8000]` | 87 | 2.88M | temporal filtering per channel |
| B1.PW | pointwise 1x1, 24->32 | `[B,24,1,8000] -> [B,32,1,8000]` | 87 | 6.14M | channel mixing |
| B2.DW | depthwise k=15, s=2 | `[B,32,1,8000] -> [B,32,1,4000]` | 199 | 1.92M | longer event context |
| B2.PW | pointwise 1x1, 32->48 | `[B,32,1,4000] -> [B,48,1,4000]` | 199 | 6.14M | expand feature bank |
| B3.DW | depthwise k=11, s=2 | `[B,48,1,4000] -> [B,48,1,2000]` | 359 | 1.06M | mid-level sound features |
| B3.PW | pointwise 1x1, 48->64 | `[B,48,1,2000] -> [B,64,1,2000]` | 359 | 6.14M | channel mixing |
| B4.DW | depthwise k=9, s=2 | `[B,64,1,2000] -> [B,64,1,1000]` | 615 | 0.58M | compact temporal map |
| B4.PW | pointwise 1x1, 64->96 | `[B,64,1,1000] -> [B,96,1,1000]` | 615 | 6.14M | higher feature capacity |
| B5.DW | depthwise k=9, s=2 | `[B,96,1,1000] -> [B,96,1,500]` | 1127 | 0.43M | high-level event abstraction |
| B5.PW | pointwise 1x1, 96->128 | `[B,96,1,500] -> [B,128,1,500]` | 1127 | 6.14M | channel expansion |
| B6.DW | depthwise k=7, s=2 | `[B,128,1,500] -> [B,128,1,250]` | 1895 | 0.22M | final downsample |
| B6.PW | pointwise 1x1, 128->160 | `[B,128,1,250] -> [B,160,1,250]` | 1895 | 5.12M | compact 160-channel representation |
| B7.DW | depthwise k=15, s=1 | `[B,160,1,250] -> [B,160,1,250]` | 5479 | 0.60M | refine event sequence, larger context |
| B7.PW | pointwise 1x1, 160->160 | `[B,160,1,250] -> [B,160,1,250]` | 5479 | 6.40M | final channel mixing |
| GAP | average over time | `[B,160,1,250] -> [B,160]` | full clip by aggregation | small | clip-level summary |
| FC | linear 160->10 | `[B,160] -> [B,10]` | full clip | 0.002M | logits |

Estimated total:

```text
Conv/FC MACs: about 61.83M MAC/clip
Conv/FC params without BN: about 78.3K
Estimated params with BN: about 81K
Final temporal stride: 256 samples = 16 ms
Final temporal length: 250 positions
Last local receptive field: 5,479 samples = about 342 ms
Global average pooling aggregates all 4 seconds
```

### Why This Shape Makes Sense

The model is intentionally structured as:

```text
early layers: preserve time resolution and capture transients
middle layers: expand channels while compressing time
late layers: refine compact event sequence
GAP: aggregate the whole 4-second sound event
```

For environmental sounds:

- `gun_shot` and `car_horn` need transient sensitivity;
- `air_conditioner`, `engine_idling`, `jackhammer` need longer texture/context;
- `children_playing`, `street_music`, `siren` need global clip-level evidence.

The model keeps 250 final time positions instead of collapsing too early, so global pooling can still see where events occur across the clip.

### Expected Tradeoff

| Model | MAC/clip | Params | Expected role |
|---|---:|---:|---|
| Efficient full-clip current | 98.7M | 149K | research baseline, better capacity |
| KV260AudioNet-DS1D proposed | 61.8M | about 81K | deployable target |
| KV260AudioNet-DS1D width 1.25 | about 95-105M | about 125K | accuracy fallback within Excellent tier |
| TCAM augmented | about 3.45B | 409K | teacher / accuracy reference |

## Training Optimization Plan

Do not train to completion and optimize later. The training loop should optimize under the deployment budget from the start.

### Phase A: Validation Protocol

Use source-independent train/validation/test:

```text
train groups: used for weight updates
validation source groups: used for model selection and augmentation/width decisions
test source groups: used only for final report
```

Current `source_group_9_1` is good for train/test comparison but not ideal for tuning because it has no validation split. Add `source_group_8_1_1` next.

### Phase B: Width Search Under MAC Budget

Run:

```text
width 0.75: target about 40-50M MAC/clip
width 1.0: target about 60-100M MAC/clip
width 1.25: target about 100M MAC/clip
```

Stop widening if validation accuracy does not improve materially.

### Phase C: Accuracy Improvements During Training

Use:

- CrossEntropy, not MSLE, for the proposed model;
- label smoothing 0.02-0.05;
- balanced class weights;
- random gain;
- time shift;
- controlled background noise;
- waveform masking;
- mixup only after baseline stability is known;
- teacher distillation from TCAM augmented if TCAM reaches much higher validation accuracy.

### Phase D: Deployment-Aware Fine-Tuning

After a model reaches useful validation accuracy:

1. Export/implement Conv2D-H1 version.
2. Run Vitis AI Model Inspector.
3. Replace unsupported operators.
4. Run INT8 PTQ.
5. If accuracy drops too much, run QAT.
6. Apply channel pruning only after accuracy is promising.
7. Fine-tune pruned model.
8. Compile and measure board latency.

## What Must Be Reported For Every Experiment

Every result must include:

```text
model name
split protocol
train accuracy
validation accuracy, if available
test accuracy
params
MAC/clip
frames/clip
augmentation settings
optimizer/loss
whether graph passed Vitis AI inspection
```

The target final statement should look like:

```text
The proposed 1D-CNN reaches X% source-independent UrbanSound8K accuracy
with Y parameters and Z MAC/clip, and its Conv2D-H1 graph is compatible
with the KV260 Vitis AI deployment flow.
```
