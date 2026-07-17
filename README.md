# TCAM1DCNN Training Project for RTX 3090 Server

Dự án này triển khai mô hình mạng nơ-ron tích chập 1 chiều kết hợp cơ chế chú ý (TCAM1DCNN) với mục tiêu tái tạo và tối ưu hóa hiệu năng mô hình nhận diện âm thanh môi trường trên tập dữ liệu UrbanSound8K, thiết kế mô-đun hóa chuyên nghiệp và tối ưu hóa tối đa hiệu năng cho cấu hình phần cứng server chứa **GPU NVIDIA RTX 3090 (24GB VRAM)**.

## 1. Cấu trúc thư mục dự án
*   `train.py`: entrypoint train chính, giữ ở root để các lệnh server hiện tại không cần đổi.
*   `configs/`: cấu hình thí nghiệm và cấu hình phần cứng/server.
*   `src/`: package nguồn chính, gồm `data/`, `models/`, `training/`, `evaluation/`, `utils/`.
*   `tools/`: script phân tích, kiểm thử reproduction và kiểm tra FLOPs.
*   `docs/architecture/`: ghi chú kiến trúc, tham số và FLOPs.
*   `docs/reproduction/`: kế hoạch tái lập, diagnostic report và kết luận thí nghiệm.
*   `docs/hardware/`: thông tin phần cứng server/workstation.

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

### Bước 3: Chạy huấn luyện (Official 10-Fold CV)
Để huấn luyện mô hình cho một fold cụ thể (ví dụ Fold 1):
```bash
python train.py --fold 1 --config configs/reproduce_msle.json --exp_name paper9_msle
```

Mặc định các config reproduce dùng protocol `paper_9_1`: train trên 9 fold và test trên 1 fold, không chọn checkpoint theo validation. Các config này cũng dùng baseline FP32 (`amp: false`), không gradient clipping (`gradient_clip: null`) và `adam_eps: 1e-7` để gần cấu hình paper hơn trước khi bật tối ưu hiệu năng. Nếu cần thí nghiệm sạch có validation riêng, gọi rõ:

```bash
python train.py --fold 1 --config configs/reproduce_msle.json --protocol clean_8_1_1 --exp_name cleanval_msle
```

Để kiểm tra đối chứng xem kết quả paper có thể đến từ random clip split thay vì official UrbanSound8K fold split hay không:

```bash
python train.py --fold 1 --config configs/random_clip_msle.json --exp_name randomclip_msle_fp32
```

Kết quả huấn luyện sẽ được lưu tự động gồm history, metrics, predictions và các snapshot checkpoint theo chu kỳ cosine.

Các thư mục `experiments/`, `checkpoints/`, `logs/`, `results/` và `data/` là artifact/local data sinh ra trong quá trình chạy, không track trong Git; phần kết luận nghiên cứu được giữ trong `docs/`.
