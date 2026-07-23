# Server Specs For Students

Tóm tắt ngắn gọn các máy GPU lab đang được nhắc tới trong dự án.

## 1. Danh Sách 3 Server GPU

| Server / máy | Trạng thái | CPU | RAM | GPU | GPU cores | VRAM |
|---|---|---|---|---|---:|---:|
| `.9` / `CPU-FPGA-GPU` | **Active chính cho train/eval** | Intel Core i7-8700K, 6 cores / 12 threads | ~32 GB installed; ~31 GiB usable | NVIDIA GeForce RTX 3090 | 10,496 CUDA cores | 24 GB |
| `.13` / RTX 3090 server | Standby, chỉ dùng khi được phân công | CPU model chưa ghi trong repo; ~28 logical CPUs | ~251 GB | NVIDIA GeForce RTX 3090 | 10,496 CUDA cores | 24 GB |
| `SERVER-02` / RTX 8000 server | Standby, ưu tiên teacher/VRAM lớn khi được phân công | CPU model/core count chưa ghi trong repo; multi-core | ~188 GB | NVIDIA Quadro RTX 8000 | 4,608 CUDA cores | 48 GB |

## 2. Server Active Hiện Tại: `.9` / `CPU-FPGA-GPU`

| Hạng mục | Thông số |
|---|---|
| Hostname | `CPU-FPGA-GPU` |
| Hệ điều hành | Ubuntu Linux |
| Kiến trúc hệ thống | x86_64 |
| CPU | Intel Core i7-8700K @ 3.70 GHz |
| CPU cores / threads | 6 physical cores / 12 logical CPUs |
| CPU turbo tối đa | 4.70 GHz |
| CPU cache | 12 MiB L3 |
| RAM | ~32 GB installed; Linux usable ~31 GiB |
| Swap | 2 GiB |
| GPU | NVIDIA GeForce RTX 3090 |
| Số GPU | 1 GPU, index `0` |
| GPU CUDA cores | 10,496 CUDA cores |
| GPU VRAM | 24 GB; reported `24576 MiB` |
| GPU memory type | GDDR6X |
| NVIDIA driver | 535.309.01 |
| CUDA supported | 12.2 |
| GPU power limit | 350 W |
| MIG | Not available |

## 3. Runtime Mặc Định Cho Server Active

| Thiết lập | Giá trị |
|---|---|
| Worktree | `~/Dung_TDTU/AI_accelerator_sound_source_tests` |
| Conda env | `sound_env` |
| CUDA device | `CUDA_VISIBLE_DEVICES=0` |
| Batch size chính | 64 |
| DataLoader workers | 6 |
| Mixed precision | `amp=true` |
| Session dài | `screen` |
| Git branch | `main` |

## 4. Ghi Chú Gửi Sinh Viên

- Server đang dùng chính cho kết quả nghiên cứu hiện tại là `.9` / `CPU-FPGA-GPU`.
- `.13` và `SERVER-02` là máy standby; không tự ý chạy job nếu chưa được phân công.
- Paper/thesis nên ghi training hardware là **NVIDIA RTX 3090**, trừ khi có run riêng dùng máy khác.
- CPU model của `.13` và `SERVER-02` chưa được ghi đầy đủ trong repo; nếu cần đưa vào báo cáo chính thức, cần chạy inventory trực tiếp trên hai máy đó.

## Sources

- Local inventory: [`Server_Hardware.md`](Server_Hardware.md)
- Server policy: [`../main/SERVER_POLICY.md`](../main/SERVER_POLICY.md)
- NVIDIA RTX 3090 specs: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
- NVIDIA Quadro RTX 8000 specs: https://www.nvidia.com/en-au/products/workstations/quadro/rtx-8000/
