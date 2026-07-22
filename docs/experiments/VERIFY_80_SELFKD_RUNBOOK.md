# Kiểm chứng lại ~80% student (Model B recipe) trên RTX 8000

## 80% là gì — không nhầm với AST

| Claim | Exp | Cách train |
|-------|-----|------------|
| **80.00%** best-val test | `local_finetune_kdprotect_f1_20ep` | **Self-KD**: student + teacher **cùng** DS1D |
| Init / teacher | `local_multifold_pyramid_base.../cycle_final` ~79% | Không phải AST |

AST-KD trên 8000 (72% / 70%) **không** phải đường 80%.

## Vì sao no-teacher 8000 trước đó chỉ ~70.7%?

Cùng SDP seed 83 fold1, server run `server8000_student_noteacher...` ≈ **70.7%** trong khi local peak ≈ **79%**.  
Variance fold1 + recipe/seed GPU có thể làm base thấp → self-KD từ base 70% **khó** nhảy lên 80%.

Kiểm chứng 80% = **lặp đúng recipe Model B (self-KD)**, không phải AST.

## Protocol kiểm chứng (2 phase, SDP f1 seed 83)

### Phase V1 — No-teacher (MAIN-like)

```bash
python tools/run_multifold.py \
  --config configs/student_ds1d_noteacher_sdp811_server_val.json \
  --exp_name server8000_verify80_noteacher_sdp811_f1_50ep \
  --folds 1 --epochs 50 \
  --data_dir data/UrbanSound8K \
  --analyze --eval_modes
```

**Gate V1:** best-val test **≥ 76%** (ideal ≥ 77–79%).  
Nếu **< 74%**: đừng kỳ vọng self-KD ra 80%; xem data/seed/log trước Phase V2.

### Phase V2 — Self-KD protect (giống Model B)

```bash
# init + teacher = cùng best.pt từ V1
python tools/run_multifold.py \
  --config configs/student_ds1d_selfkd_verify80_sdp811_val.json \
  --exp_name server8000_verify80_selfkd_sdp811_f1_20ep \
  --folds 1 --epochs 20 \
  --data_dir data/UrbanSound8K \
  --analyze --eval_modes
```

Config: λ=**0.6**, T=**2**, protect_classes **[1,4,6,7,8]**, lr=**2e-4**, apply_to_mixup **true**  
(khớp `kv260_ds1d_pyramid_finetune_weakboost_kdprotect_val.json`).

**Success:** best-val test **≥ 78%** (soft verify); **≥ 80%** = full match local claim.  
**Soft pass:** +1–2 pp so với V1 best (self-KD hoạt động).  
**Fail:** ≤ V1 → self-KD không giúp trên server.

## Cách khác nhanh (nếu còn file local 79/80)

Copy lên 8000:

```text
# student init / teacher
local_multifold.../tcam_fold_1_cycle_final.pt  hoặc  tcam_fold_1_best.pt
```

Sửa template trong config trỏ path đó, chỉ chạy Phase V2 20ep — kiểm **re-finetune** có ~80% không (gần reproduce nhất).

## Không dùng AST trong verify 80%

Teacher = **DS1D live** `build_model` + `.pt`, không `cached_logits` AST.
