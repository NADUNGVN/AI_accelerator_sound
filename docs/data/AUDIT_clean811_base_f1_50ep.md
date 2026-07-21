# Audit: `server3090_clean811_base_f1_50ep` (best-val predictions)

**Protocol:** `clean_8_1_1` — train folds 3–10, val fold 2, test fold 1  
**Model:** `kv260_audio_net_ds1d`, 50 ep, seed 83  
**Git:** `origin/main` metrics @ `79537b6`  
**Primary predictions:** `best_val_model_predictions` (873 clips)

## 1. Headline

| Metric | Value |
|---|---:|
| Best val | 63.18% |
| **Best-val test** | **65.06%** (568/873) |
| Last snapshot | 65.29% |
| Ensemble | 66.44% |
| Errors | **305** |
| To reach **80%** | need **≥699** correct → **+131** more correct clips |
| To reach **79.08%** (old peak, different protocol) | **+123** clips (not comparable protocol) |

Selection is consistent (best-val / last / ens within ~1.4 pp) — unlike 100ep+ES collapse.

## 2. Per-class (best-val model)

| Class | Support | Acc | Main confusions |
|---|---:|---:|---|
| gun_shot | 35 | **97.1%** | (near saturated) |
| car_horn | 36 | **91.7%** | |
| siren | 86 | 83.7% | |
| dog_bark | 100 | 81.0% | |
| children_playing | 100 | 74.0% | |
| drilling | 100 | 69.0% | |
| street_music | 100 | 61.0% | → children, AC |
| air_conditioner | 100 | **50.0%** | → street_music 17, jackhammer 16, engine 8 |
| engine_idling | 96 | **44.8%** | → **AC 40**, jackhammer 8 |
| jackhammer | 120 | **42.5%** | → **drilling 36**, **engine 30** |

**Max theoretical gain if one class became perfect (overall pp):**

| Class | Miss | Max +pp |
|---|---:|---:|
| jackhammer | 69 | **7.90** |
| engine_idling | 53 | **6.07** |
| air_conditioner | 50 | **5.73** |
| street_music | 39 | 4.47 |
| drilling | 31 | 3.55 |

→ Room to **>80%** must come mainly from **industrial / continuous-noise family** (jack + engine + AC), not gun/horn.

## 3. Top confusions (true → pred)

| Pair | Count |
|---|---:|
| engine_idling → air_conditioner | **40** |
| jackhammer → drilling | **36** |
| jackhammer → engine_idling | **30** |
| air_conditioner → street_music | 17 |
| air_conditioner → jackhammer | 16 |
| street_music → children_playing | 11 |

**Essence:** one **machinery / continuous texture** cluster  
`{air_conditioner, engine_idling, jackhammer, drilling}`  
plus ambient `{street_music, children_playing}`.

## 4. Source (fsID) concentration

| | |
|---|---|
| Top-5 fsID | **45.2%** of all errors |
| Top-10 fsID | **57.0%** |
| Top-15 fsID | **66.2%** |
| Top-20 fsID | **73.8%** |

### Highest-error sources

| fsID | Errors | Pattern |
|---|---:|---|
| **180937** | 61/79 (77%) | jackhammer → engine (30) + drilling (30) |
| **176787** | 31/31 (**100%**) | engine → AC (26) |
| **134717** | 25/25 (**100%**) | AC → street (17), jackhammer (8) |
| 103258 | 13/20 | engine → AC |
| 119455 | 8/8 | engine → jackhammer |
| 59277 | 8/8 | AC → engine |

**Whole sources fully wrong (support ≥3):** e.g. 176787 (31), 134717 (25), 59277 (8), 119455 (8), …

→ Failures are **not random clip noise**; they are **source-level collapses** inside the industrial cluster.

## 5. Causal reading (for algorithm design)

1. Under held-out fold1, test sources are **unseen groups**.  
2. Model confuses **same acoustic family** across sources (jack/drill/engine/AC).  
3. A few fsIDs dominate errors → representation is **source-conditioned** inside the machinery manifold.  
4. +131 clips to 80% is large: need systematic fix of that manifold, not epoch tuning.

## 6. Implications (next design, not yet experiment)

Priority loss/representation targets (data-driven from this audit):

1. **jackhammer ↔ drilling ↔ engine_idling**  
2. **engine_idling ↔ air_conditioner**  
3. Optional: AC ↔ street_music  

Algorithm families that match this structure (for later implementation):

- Group-robust / source-group difficulty on **train** sources in machinery classes  
- Same-class, cross-source pull **only** inside machinery labels  
- Confusion-pair loss **only** on pairs above (narrow), after one-axis ablations  

**Out of scope here:** KD, ES-by-val, broad hard-neg industrial soup (already failed under other protocol).

## 7. Dataset scope for paper (locked add-ons)

See workspace `EXTRA_DATASETS_FOR_PAPER.md` — **ESC-50 + Speech Commands** locked as secondary benchmarks; primary remains US8K clean811.
