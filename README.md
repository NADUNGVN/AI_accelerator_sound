# TCAM1DCNN Training Project for RTX 3090 Server

Dự án này triển khai mô hình mạng nơ-ron tích chập 1 chiều kết hợp cơ chế chú ý (TCAM1DCNN) với mục tiêu tái tạo và tối ưu hóa hiệu năng mô hình nhận diện âm thanh môi trường trên tập dữ liệu UrbanSound8K, thiết kế mô-đun hóa chuyên nghiệp và tối ưu hóa tối đa hiệu năng cho cấu hình phần cứng server chứa **GPU NVIDIA RTX 3090 (24GB VRAM)**.

## 1. Cấu trúc thư mục dự án
*   `configs/`: Chứa tệp cấu hình phần cứng `rtx3090_config.json`.
*   `src/models/`: Định nghĩa mô hình mạng `TCAM1DCNN`, Time Attention Module (TAM) và Channel Attention Module (CAM).
*   `src/data/`: Xử lý tập dữ liệu, nạp âm thanh thô, resample 16kHz và RAM Caching.
*   `src/training/`: Mã nguồn lớp `Trainer` điều khiển vòng lặp huấn luyện, mixed precision và đánh giá.
*   `src/utils/`: Chứa các hàm phụ trợ thiết lập seed ngẫu nhiên và chuẩn bị thư mục.
*   `train.py`: Tệp kích hoạt chạy huấn luyện chính từ dòng lệnh.

## 2. Điểm tối ưu hóa đặc biệt dành riêng cho RTX 3090 (24GB VRAM)
1.  **Batch Size trực tiếp:** Vì RTX 3090 sở hữu dung lượng VRAM lớn lên tới 24 GB, chúng ta thiết lập kích thước lô vật lý trực tiếp bằng **96** (`batch_size: 96`), loại bỏ sự cần thiết của Gradient Accumulation như trên các dòng card laptop (RTX 5060/4060). Điều này giúp tối đa hóa khả năng tính toán song song của Tensor Cores trên GPU.
2.  **RAM Caching:** Toàn bộ 8,732 file âm thanh sau khi giải mã và resample về 16kHz (mono, dài 4s) chỉ chiếm khoảng **2.2 GB RAM**. Server của chúng ta có 32 GB RAM, do đó toàn bộ tập dữ liệu được lưu trực tiếp trên RAM vật lý. Việc này loại bỏ hoàn toàn hiện tượng nghẽn cổ chai đọc đĩa cứng (Disk I/O bottleneck) và giúp tốc độ train tăng đột biến từ **50 đến 100 lần**.
3.  **Tự động hóa Mixed Precision (AMP):** Huấn luyện ở chế độ FP16 tự động giúp GPU tính toán cực nhanh và tiết kiệm băng thông bộ nhớ.

## 3. Hướng dẫn cài đặt và sử dụng

### Bước 1: Cài đặt thư viện
Cài đặt các gói phụ thuộc từ tệp `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu
Hiện tại tập dữ liệu đang được lưu trữ sẵn tại thư mục:
```text
data/raw/
└── UrbanSound8K/
    ├── metadata/
    │   └── UrbanSound8K.csv
    └── audio/
        ├── fold1/
        ├── fold2/
        ...
```
*(Đường dẫn mặc định trong mã nguồn đã được cấu hình trỏ thẳng tới `data/raw/UrbanSound8K`)*.

### Bước 3a: Reproduce paper Abdoli et al. 2019 (khuyến nghị)

Config bám sát paper (Table 1 + Table 3 best 89%):
- Model: `Abdoli1DCNN` variant `gamma` (CL1 Gammatone 64×512, **frozen**)
- Input: 16 000 samples (1 s @ 16 kHz), hop 8000 (**50% overlap**), rectangular window
- Framing: **độ dài clip thật** (`pad_to_seconds=null`), bỏ frame gần im lặng; không ép canvas 4 s zero như pipeline TCAM cũ
- Loss: MSLE (Eq. 4) · Optimizer: Adadelta lr=1.0 · batch=100 · epochs≤100 + early stop
- Protocol: `clean_8_1_1` (8 train / 1 val / 1 test, official US8K folds)
- Aggregation: **sum rule** trên softmax của các frame

```bash
# Best paper setup (reported 89%) — one fold
python train.py --fold 1 --config configs/paper_abdoli_gamma.json --exp_name paper_abdoli_gamma

# Random-init setup (reported 87%)
python train.py --fold 1 --config configs/paper_abdoli_rand.json --exp_name paper_abdoli_rand

# Full 10-fold CV (gamma) — recommended on server
bash scripts/run_paper_abdoli_10fold.sh

# Resume from fold 2 if fold 1 already done
bash scripts/run_paper_abdoli_10fold.sh --start_fold 2 --end_fold 10

# Aggregate mean ± std
python scripts/summarize_10fold.py --exp_name paper_abdoli_gamma
```

Metric chính để so với paper: **Mean test (best-val model)** từ `summarize_10fold.py` / từng `metrics.json` field `test_acc_best_val_model`.

### Bước 3b: TCAM / research baselines

```bash
python train.py --fold 1 --config configs/reproduce_msle.json --exp_name paper9_msle
python train.py --fold 1 --config configs/reproduce_msle.json --protocol clean_8_1_1 --exp_name cleanval_msle
python train.py --fold 1 --config configs/random_clip_msle.json --exp_name randomclip_msle_fp32
```

Kết quả được lưu dưới `experiments/<exp_name>/fold_<k>/` (history, metrics, predictions, checkpoints).

Các thư mục `experiments/`, `checkpoints/`, `logs/`, `results/` và `data/` là artifact/local data sinh ra trong quá trình chạy, không track trong Git; phần kết luận nghiên cứu được giữ trong `docs/`.
