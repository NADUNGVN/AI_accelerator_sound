# Tools

Các script trong thư mục này là công cụ kiểm thử và phân tích, không phải package runtime chính.

- `analyze_experiment.py`: đánh giá checkpoint/ensemble và xuất per-class accuracy, confusion, aggregation mode.
- `research_diagnostics.py`: kiểm tra sâu dữ liệu, split, kiến trúc, tham số và MACs.
- `flops_lower_bound_check.py`: kiểm tra lower-bound MACs/FLOPs từ các lớp Conv1D trong paper.
- `export_layer_q16_txt.py`: xuất checkpoint thành từng file `*_weight_q16.txt` / `*_bias_q16.txt` theo layer, mặc định Conv+BN fused.

Chạy từ root repo, ví dụ:

```bash
python tools/analyze_experiment.py --help
python tools/research_diagnostics.py --help
python tools/flops_lower_bound_check.py
python tools/export_layer_q16_txt.py --export_deploy_models
```
