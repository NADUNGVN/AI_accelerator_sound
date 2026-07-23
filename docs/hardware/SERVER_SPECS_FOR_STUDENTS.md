# Server Specs For Students

**Server chính dùng cho train/eval:** `CPU-FPGA-GPU`

| Hạng mục | Thông số |
|---|---|
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

## Training Runtime

| Thiết lập | Giá trị mặc định |
|---|---|
| Conda env | `sound_env` |
| CUDA device | `CUDA_VISIBLE_DEVICES=0` |
| Batch size chính | 64 |
| DataLoader workers | 6 |
| Mixed precision | `amp=true` |

## Ghi Chú

- Đây là server mặc định cho các run chính trong Phase A và Phase 1 dataset.
- Local Windows chỉ dùng để sửa code, tài liệu, và smoke test nhỏ.
- Các máy `.13`, RTX 8000 / `SERVER-02`, và KV260 chỉ dùng khi được phân công riêng.
- Số CUDA cores là thông số chuẩn của NVIDIA GeForce RTX 3090; các thông số driver, VRAM, CPU, RAM là từ inventory server hiện có trong repo.

## Sources

- Local inventory: [`Server_Hardware.md`](Server_Hardware.md)
- Server policy: [`../main/SERVER_POLICY.md`](../main/SERVER_POLICY.md)
- NVIDIA RTX 3090 specs: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
