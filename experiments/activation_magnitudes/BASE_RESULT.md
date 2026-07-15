# Qwen3-4B-Base activation magnitude 实验结果

> 执行日期：2026-07-11  
> 模型：`models/Qwen3-4B-Base`，BF16  
> 原始数据：[`results_base/`](results_base/)

Base 与 Instruct 使用统一的 9 个 benchmark、每项固定采样 16 条，共 144 prompt / 16,132 token。
Base 主量级探针完整结束。

| 指标 | Qwen3-4B-Base | Qwen3-4B-Instruct |
|---|---:|---:|
| logits RMS | 8.67 | 3.77 |
| logits mean abs | 7.63 | 2.92 |
| prompt median top1-top2 margin | 1.44 | 2.56 |
| L0 intrinsic BF16 GEMM rel error | 2.17e-3 | 2.20e-3 |
| L17 intrinsic BF16 GEMM rel error | 2.32e-3 | 2.34e-3 |
| L35 intrinsic BF16 GEMM rel error | 2.30e-3 | 2.26e-3 |

尽管 Base 与后训练模型的 activation/logits 尺度不同，`down_proj` BF16 GEMM 的固有相对
误差仍稳定在约 `2.2–2.3e-3`，支持有限精度机制不依赖后训练版本的结论。

`results/down_gemm_decomposition.json` 没有对应的生成脚本，因此 Base 结果不包含这一项。
当前受版本控制的 `probe_magnitudes.py` 所能生成的 Base 产物均已保存。

```bash
cd experiments/activation_magnitudes
CUDA_VISIBLE_DEVICES=1 conda run -n qwen3 python probe_magnitudes.py \
  --model-path ../../../models/Qwen3-4B-Base --results-dir results_base
```
