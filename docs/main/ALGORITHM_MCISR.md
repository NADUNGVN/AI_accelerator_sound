# Algorithm: MC-ISR  
## Machinery-Cluster + Source-Robust training (no KD, no ES)

**Driven by:** audit `server3090_clean811_base_f1_50ep`  
**Protocol:** `clean_8_1_1` (test fold1, val fold2, train 3–10)  
**Backbone:** unchanged `kv260_audio_net_ds1d` full-clip  

---

## 1. Problem statement (from audit)

On 873 fold1 test clips (best-val model, 65.06%):

- **305 errors**; need **+131** correct for 80%.
- Errors concentrate in **machinery / continuous-noise cluster**:  
  `{air_conditioner=0, drilling=4, engine_idling=5, jackhammer=7}`.
- Top confusions: `engine→AC (40)`, `jack→drill (36)`, `jack→engine (30)`.
- **Top-5 fsID = 45%** of errors; several sources **100% wrong** (e.g. 176787, 134717).

**Essence:** under held-out sources, the encoder maps whole **source groups** of machinery classes into the wrong region of the machinery manifold.

---

## 2. What failed before (do not repeat)

| Method | Why not reuse as-is |
|---|---|
| Broad hard-neg + multi-HP (T2) | Confounded; jackhammer collapse |
| AC–engine only HN (H2) on source_group | Val–test mess; did not reduce AC–engine |
| Source-aware batch only (T1) | Hurt best-val test |
| Val early-stopping long train | Locked bad checkpoints |

MC-ISR differs: **audit-narrow pairs only** + **explicit source-group robust term on train sources** + **clean811 protocol** + **one config, no ES**.

---

## 3. Algorithm

### Notation

- Batch \(\{(x_i, y_i, s_i)\}\): waveform, class, **train source id** \(s_i\) (from fsID+class on train only).
- Machinery set \(\mathcal{M}=\{0,4,5,7\}\).
- Audit directed pairs \(\mathcal{P}\) (from clean811 top confusions):

```text
(5,0), (0,5)     engine ↔ AC
(7,4), (4,7)     jackhammer ↔ drilling
(7,5), (5,7)     jackhammer ↔ engine
```

(No jackhammer↔AC pair: not top pattern on this audit; avoid T2-style over-coupling.)

### Loss

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda_{\mathrm{pair}}\,\mathcal{L}_{\mathrm{pair}}
+
\lambda_{\mathrm{src}}\,\mathcal{L}_{\mathrm{src}}
\]

**1) \(\mathcal{L}_{\mathrm{CE}}\)**  
Standard cross-entropy (existing class weights; optional mild boost on \(\mathcal{M}\)).

**2) \(\mathcal{L}_{\mathrm{pair}}\) — cluster margin (logit)**  
For each \((c,n)\in\mathcal{P}\), on samples with \(y_i=c\):

\[
\ell_{c\leftarrow n}
=
\mathrm{ReLU}\big(z_{i,n}-z_{i,c}+m\big)
\]

Average over active pairs in the batch.  
Default: \(m=0.25\), \(\lambda_{\mathrm{pair}}=0.02\), **not** applied under mixup (same as safe H2 setting).

**Why:** directly penalizes the observed swaps inside the machinery manifold.

**3) \(\mathcal{L}_{\mathrm{src}}\) — source-group robust CE (machinery only)**  
Let \(I_{\mathcal{M}}=\{i: y_i\in\mathcal{M}\}\).  
Per-sample CE \(\ell_i\). For each train source \(s\) present in \(I_{\mathcal{M}}\):

\[
L_s = \mathrm{mean}\{\ell_i: i\in I_{\mathcal{M}}, s_i=s\}
\]

Smooth worst-source (stable vs hard max):

\[
\mathcal{L}_{\mathrm{src}}
=
\tau\log\sum_{s}\exp(L_s/\tau)
\quad(\tau=0.1)
\]

Default \(\lambda_{\mathrm{src}}=0.15\).

**Why:** whole-fsID collapses (176787, 134717, …) mean **group** risk, not i.i.d. clip noise. This is Group-DRO-style pressure on **train** sources only (no test fsID used).

### What is *not* in v1

- No teacher / KD  
- No early stopping  
- No source-aware batch sampler  
- No broad SupCon  

---

## 4. Training protocol (experiment)

| Item | Value |
|---|---|
| Config | `configs/kv260_ds1d_pyramid_mixup_ema_clean811_mcisr_val.json` |
| Exp name | `server3090_clean811_mcisr_f1_50ep` |
| Epochs | 50 |
| Compare to | `server3090_clean811_base_f1_50ep` (**65.06%** best-val test) |
| Gate | best-val test **> 65.06%**; jackhammer acc **≥ 42%** (no collapse); ideally AC/engine/jack confusions ↓ |

### Ablation ladder (if v1 mixed)

1. CE only (baseline already)  
2. CE + pair only (`λ_src=0`)  
3. CE + src only (`λ_pair=0`)  
4. Full MC-ISR  

---

## 5. Expected paper story

> Errors on official fold-1 under clean 8/1/1 concentrate in a machinery acoustic cluster and in a few recording sources. We introduce MC-ISR: a dual regularizer that (i) separates audited within-cluster confusions in logit space and (ii) robustifies CE to hard **training** sources inside that cluster, without teachers or test leakage.

---

## 6. After MC-ISR (not now)

Loaders: **ESC-50**, **Speech Commands** for multi-dataset tables.
