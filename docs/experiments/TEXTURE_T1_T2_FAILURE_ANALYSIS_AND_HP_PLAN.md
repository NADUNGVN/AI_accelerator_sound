# Phân tích thất bại T1/T2 + kế hoạch hyperparameter có kiểm chứng

Ngày: 2026-07-21  
Model khóa: `kv260_audio_net_ds1d` full-clip (~101.7k params), **no-teacher**  
Protocol: `source_group_8_1_1` / `fsid_classid_balanced_v1`, seed 83  
Artifacts: `results/server3090-notacher-f1`, `results/server3090-texture-f1`

## 1. “Fail” nghĩa là gì?

Không phải model “không học”. Fail theo **tiêu chí chốt model đã thống nhất**:

| Tiêu chí | Baseline | T1 sourcebalance | T2 sourcehard | Ai thắng |
|---|---:|---:|---:|---|
| **Best-val → test** (selection) | **76.90%** | 73.56% | 74.60% | **Baseline** |
| Best val | **72.52%** | 70.44% | 71.36% | Baseline |
| Ensemble | 75.40% | **79.43%** | 77.01% | T1 (metric phụ) |
| Worst final class | AC 65% | AC 70% | **jackhammer 61%** | T1 cải worst; T2 phá worst |

**Kết luận fail:** T1/T2 **không thay được baseline** nếu chọn checkpoint theo validation.  
**Không fail hoàn toàn:** có tín hiệu đổi confusion (texture), ensemble T1 tăng — can thiệp **có tác dụng**, nhưng **thiết kế & hyperparameter** làm lệch selection / đánh đổi class.

---

## 2. Class id (dùng khi đọc hard-neg pairs)

| id | class |
|---:|---|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

---

## 3. Root cause — vì sao fail (theo tầng)

### 3.1 Tầng thí nghiệm: T2 là **confounded multi-change** (lỗi thiết kế)

So với baseline, **T1** chỉ bật thêm:

- `source_aware_batch_sampler.enabled = true`  
  (`classes_per_batch=8`, `sources_per_class=2`, `samples_per_source=4`, batch 8×2×4=64)

**T2** đổi **cùng lúc** nhiều trục:

| Trục | Baseline | T2 |
|---|---|---|
| source_aware | off | on + **class_multipliers lệch mạnh** (AC 1.8, engine 1.6, gun 0.85, jack 1.35…) |
| hard_negative | off | on, weight 0.035, margin 0.35, **14 hướng pair** |
| class_weight_multipliers | [1.05,1.3,…] jack 1.45 | profile khác |
| mixup.prob | **0.70** | **0.40** |
| augment | gain 5 / noise 0.25 | gain 6 / noise 0.30 / mask mạnh hơn |

→ Không thể quy “T2 fail vì hard-neg” hay “vì source-aware”.  
Đây là **failure of experimental control**, không chỉ failure of idea.

### 3.2 Tầng tối ưu: val–test **misalignment** (selection fail)

| Run | Best val ep | Best val | Best-val test | Ensemble |
|---|---:|---:|---:|---:|
| Baseline (hist 200, đỉnh ~46) | 46 | 72.52% | **76.90%** | 75.40% |
| T1 | 35 | 70.44% | 73.56% | **79.43%** |
| T2 | 48 | 71.36% | 74.60% | 77.01% |

Giải thích:

1. **Source-aware batch** làm phân phối gradient theo source/class khác IID-ish của val set → **val accuracy thấp hơn / kém tin cậy** làm proxy test.  
2. T1: train acc max chỉ ~61% (mixup + sampling) vs baseline ~76% max — underfitting tương đối trên CE, nhưng **ensemble cuối cao** → representation cuối run khác, val sớm không chọn được.  
3. Protocol “chọn theo best val” vì vậy **trừng phạt** T1/T2 dù một số metric cuối (ensemble) đẹp hơn.

**Fail selection ≠ fail representation hoàn toàn.**

### 3.3 Tầng loss geometry: hard-neg margin **đẩy nhầm hướng**

Loss (trainer): với mỗi pair `(target, negative)`:

`ReLU(logit_neg − logit_target + margin)` trên sample thuộc `target`.

T2 pairs gồm:

- Có ích tiềm năng: `(0,5)(5,0)` AC↔engine; `(9,2)(2,9)` street↔children  
- **Nguy hiểm:** `(0,7)(7,0)` AC↔jackhammer; `(4,7)(7,4)` drilling↔jackhammer; `(0,4)(4,0)` AC↔drilling  

Evidence confusion **final cycle**:

| Run | Top confusion | Ý nghĩa |
|---|---|---|
| Baseline | AC→engine 19; street→children 16; engine→drilling 16; jack→drilling 10 | Họ texture đúng giả thuyết |
| T1 | AC→engine **22** (chưa hết); engine→drilling **giảm**; ambient swap còn | Source-aware không đủ tách AC–engine; bớt industrial spill một phần |
| T2 | **jackhammer→AC: 28**; AC→engine **25** | Hard-neg **không** giảm AC–engine; **phá jackhammer** (worst 61%) |

Cơ chế hợp lý:

- Ép jackhammer **xa** drilling/AC trên logit → model học “jackhammer không phải drilling” bằng cách **đẩy sang AC/ambient** (khoảng cách logit), không học feature bất biến source.  
- Margin loss **không** có term “giữ đúng class khác”; chỉ nới cặp chỉ định → **waterbed effect** (kéo một confusion, phồng confusion khác).  
- `weight=0.035` cộng CE + label smoothing + mixup: đủ mạnh để lệch boundary industrial (jack/drill/AC cùng họ máy).

### 3.4 Tầng sampling: source-aware **đổi empirical prior**

Batch cố định 8 class × 2 source × 4 sample:

- **Ưu:** đa dạng source trong batch → giảm “một fsID thống trị gradient” (đúng hướng texture/source).  
- **Hại:**  
  - Class hiếm / source ít bị **oversample** hoặc **lặp** → noise.  
  - Val set vẫn theo tần suất tự nhiên → **train distribution ≠ val** → best-val kém.  
  - T2 còn `class_multipliers` sampling (AC 1.8, engine 1.6) → **over-emphasize** đúng class đang confuse, dễ overfit pattern giả trên val-proxy.

### 3.5 Tầng so sánh epoch: baseline metrics 200 ep vs T1/T2 50 ep

- Baseline **đỉnh val ~ep 46** → so 50 ep **khá công bằng cho peak selection**.  
- Nhưng baseline **ensemble/last** hưởng snapshot muộn hơn (hist 200).  
- T1 ensemble cao trong **50 ep** là tín hiệu thật; đừng dùng baseline ensemble 200-ep cycle muộn để “đè” T1 không điều kiện — khi report, ghi rõ horizon.

### 3.6 Tóm tắt nhân quả (1 sơ đồ)

```text
Mục tiêu: representation bớt bám source, bớt AC↔engine
        │
        ├─ T1: chỉ source-aware batch
        │     → đổi train dist → val↓ → best-val test↓ (−3.3)
        │     → ensemble↑ (+4) : tín hiệu dương nhưng selection fail
        │
        └─ T2: source-aware + hard-neg rộng + reweight + mixup↓ + aug↑
              → confound
              → jackhammer sập (pair industrial)
              → AC↔engine không hết (vẫn top confusion)
              → best-val test vẫn < baseline
```

---

## 4. Bài học phương pháp (research hygiene)

1. **Một lần chỉ một trục** (hoặc factorial nhỏ có control).  
2. Metric chính **cố định trước**: best-val test; phụ: ensemble, worst-class, top-2 confusions chỉ định.  
3. Hard-neg pairs **chỉ từ confusion matrix baseline**, không nhét cả họ industrial.  
4. Cấm stack “source-aware + hard-neg + reweight + mixup + aug” trong 1 shot.  
5. Gate: chỉ full-fold khi fold1 best-val test **≥ baseline − 0.3 pp** và worst-class **≥ baseline worst − 2 pp**.

---

## 5. Rà tham số — bản đồ độ nhạy (hypothesis)

| Tham số | Vai trò | Giả thuyết sau T1/T2 | Ưu tiên tune |
|---|---|---|---|
| `source_aware_batch_sampler` on/off | Đa dạng source | Có tín hiệu ensemble; hại val-proxy | **Cao** — giữ nhưng chỉnh nhẹ |
| `sources_per_class` / `samples_per_source` | Cấu trúc batch | 2×4 có thể quá cứng | Trung bình |
| `hard_negative.pairs` | Hướng đẩy logit | Pairs industrial = độc | **Cao** — thu hẹp AC–engine (± street–children) |
| `hard_negative.weight` | Cường độ | 0.035 có thể mạnh khi + CE | **Cao** — 0.01–0.02 |
| `hard_negative.margin` | Độ nới | 0.35 | Trung — 0.2–0.3 |
| `mixup.prob` | Regularize | T2 hạ 0.4 confound | **Không** tune chung hard-neg; giữ 0.7 như baseline khi ablating HN |
| `class_weight_multipliers` | CE prior | T2 đổi + sampling multiplier = double count | Giữ baseline khi ablating |
| `augment.*` | Texture noise | T2 mạnh hơn = confound | Giữ baseline khi ablating |
| `lr` / `epochs` / `ema.decay` | Tối ưu chung | Chưa có bằng chứng là root fail T1/T2 | **Thấp** vòng này |
| Architecture / teacher | Capacity | Ngoài scope branch | **Cấm** vòng này |

---

## 6. Kế hoạch verify có kiểm chứng (ablation)

### 6.1 Control

| ID | Mô tả | Config |
|---|---|---|
| **H0** | Baseline full-clip (đã có server) | `kv260_ds1d_pyramid_mixup_ema_val.json` |
| **H1** | + source-aware only (đã chạy = T1) | `kv260_ds1d_pyramid_sourcebalance_ce_val.json` |

### 6.2 Vòng mới (một trục / một stack tối thiểu)

| ID | Giả thuyết | Thay đổi duy nhất (so H0 hoặc H1) | Config mới |
|---|---|---|---|
| **H2** | Hard-neg **chỉ** AC↔engine giúp confusion mà không phá jackhammer | H0 + HN pairs `(0,5)(5,0)`, w=0.02, m=0.30; **không** source-aware | `configs/kv260_ds1d_pyramid_hneg_ac_engine_val.json` |
| **H3** | HN an toàn + source-aware (stack tối thiểu có kiểm soát) | H1 + HN như H2 | `configs/kv260_ds1d_pyramid_sourcebalance_hneg_ac_engine_val.json` |
| **H4** | T2 “sạch”: bỏ pair industrial, trả mixup/aug/weights về H0 | source-aware multipliers **uniform 1.0**; HN pairs chỉ AC–engine + street–children; mixup 0.7; aug baseline | `configs/kv260_ds1d_pyramid_sourcehard_v2_safe_val.json` |

### 6.3 Protocol chạy (bắt buộc giống nhau)

```text
folds: 1
epochs: 50
seed: 83
protocol: source_group_8_1_1
model: kv260_audio_net_ds1d (không đổi)
teacher/KD: OFF
analyze + eval_modes: ON (sau train)
exp_name: server3090_texture_{h2|h3|h4}_f1_50ep
```

### 6.4 Tiêu chí quyết định (pre-registered)

Gọi \(B = 76.90\%\) best-val test baseline.

| Kết quả fold1 | Quyết định |
|---|---|
| best-val test ≥ \(B\) và worst-class ≥ 63% | **Win** — giữ hướng, cân nhắc folds 1–3 |
| best-val test ∈ \([B-1.0, B)\) và AC→engine confusion giảm ≥20% relative | **Partial** — tinh chỉnh weight/margin nhỏ, không stack thêm |
| best-val test < \(B-1.0\) hoặc worst-class < 60% | **Reject** config; không full-10 |
| Ensemble ↑ nhưng best-val test ↓ (như T1) | Ghi nhận misalignment; thử **selection** (EMA eval, last-5 mean val) — **không** tự tuyên bố SOTA |

### 6.5 Thứ tự chạy (tiết kiệm GPU)

1. **H2** (rẻ về ý nghĩa nhân quả: HN thuần)  
2. Nếu H2 không phá jackhammer → **H3**  
3. **H4** chỉ nếu muốn “cứu” ý tưởng T2 đã confounded  
4. Không chạy H4 trước H2/H3

---

## 7. Lệnh server (sau khi pull research)

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
git pull origin research/fpga-1dcnn-90acc
conda activate sound_env
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p logs/texture_hp

# H2
nohup python tools/run_multifold.py \
  --config configs/kv260_ds1d_pyramid_hneg_ac_engine_val.json \
  --exp_name server3090_texture_h2_hneg_ac_engine_f1_50ep \
  --folds 1 --epochs 50 --analyze --eval_modes \
  > logs/texture_hp/h2.nohup.log 2>&1 &
```

Tương tự H3/H4 với config/exp_name tương ứng.  
Push results: `git add -f experiments/.../metrics.json history.json predictions.json analysis_all_cycles.json multifold_summary.*`

---

## 8. Kết luận ngắn

1. **Fail selection** vì source-aware làm **val kém là proxy**; ensemble T1 cho thấy can thiệp **không vô nghĩa**.  
2. **Fail T2 class** vì hard-neg **quá rộng** (jackhammer–AC/drill) + **nhiều HP đổi cùng lúc**.  
3. Hướng nghiên cứu đúng: **ablation một trục**, HN **chỉ AC–engine** (± street–children), giữ mixup/aug/CE weight như baseline.  
4. Không tăng lr/architecture/teacher ở vòng này — chưa phải root cause đã chứng minh.
