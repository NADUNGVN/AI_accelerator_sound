# Data analysis (canonical)

Phân tích **dữ liệu + kết quả thí nghiệm** theo chuẩn này trước khi claim cải tiến hoặc promote lên `main`.

| Doc | Nội dung |
|---|---|
| [DATASET_PROFILE.md](DATASET_PROFILE.md) | UrbanSound8K: quy mô, class, fold, fsID, ràng buộc |
| [SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md) | `source_group_8_1_1` vs official `paper_9_1` vs random |
| [MULTIDATASET_PHASE1.md](MULTIDATASET_PHASE1.md) | Phase 1 contract + loader/config cho ESC-50 + Speech Commands trước khi claim |
| [ANALYSIS_STANDARD.md](ANALYSIS_STANDARD.md) | Checklist phân tích **mỗi** experiment (metrics → confusion → quyết định) |
| [RESULT_REPORT_TEMPLATE.md](RESULT_REPORT_TEMPLATE.md) | Mẫu báo cáo 1 run (copy-paste) |
| [AUDIT_clean811_base_f1_50ep.md](AUDIT_clean811_base_f1_50ep.md) | Audit baseline clean811 (class × fsID) |

## Related artifacts (do not lose)

| Location | What |
|---|---|
| Workspace `docs_workspace/30_dataflow_split/` | Silent-frame, split-effect write-ups |
| Workspace `docs_workspace/10_baseline_results/` | Server H0 analyses |
| `docs/notebooks/UrbanSound8K_1D_CNN_Dataflow_Research.ipynb` | Dataflow WAV → tensor |
| `tools/analyze_experiment.py` | Cycle eval, confusion, modes |
| `tools/run_multifold.py` | Multifold summary JSON/MD |
| Git `results/*` branches | Pushed metrics (not full `.pt`) |

## Multi-dataset Phase 1 scope (locked)

| Priority | Dataset | Role |
|---|---|---|
| 1 (primary) | **UrbanSound8K** `clean_8_1_1` | Main method + audit |
| 2 | **ESC-50** | ESC comparability (50-class, 5-fold) |
| 3 | **Speech Commands** (subset) | Edge / short-clip / KV260 story |

Run order and split contracts: [MULTIDATASET_PHASE1.md](MULTIDATASET_PHASE1.md).

## Rule

No new training claim without:

1. Protocol + seed + config recorded  
2. Primary metric filled (main path: **best-val test**)  
3. Overlap source train–test checked (US8K)  
4. Confusion / worst-class noted  
5. Compared to clean811 baseline or ACHIEVED peak as appropriate  
