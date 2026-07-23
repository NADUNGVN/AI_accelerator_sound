# Per-layer Q16 text export

Source checkpoint: `D:\Research\Research_Development_System\5_Thesis_Optimization_Guides\1_ai_accelerator_sound\repo\Dung_TDTU\1_ai_accelerator_sound_mainline\deploy\student_models\model_b_kd_student_80p00\model_full.pt`
Mode: `bn_fused`
Name style: `sequential`
Number format: `decimal`

Each layer has two text files: `*_weight_q16.txt` and `*_bias_q16.txt`.
Values are one signed INT16 word per line unless `number_format=hex` was used.
Use `manifest_q16.json` for tensor shapes, scales, source tensor names, and BN-fold details.

Float reconstruction per tensor:

```text
float_value_approx = q16_value * scale_float_equals_q_times_scale
```
