# Qwen3-4B-Base 实验结果

> 执行日期：2026-07-11  
> 模型：`models/Qwen3-4B-Base`，BF16，36 层  
> 原始数据：[`results_base/`](results_base/)

Qwen3-4B-Base 完成了 A/B/C 三阶段实验。所有代数单测、坐标对齐、canonical-down、
确定性和权重恢复检查均通过。

## 核心结果

- Stage A：通过，0 个问题。
- Stage B：159 个 case 全部完成，restore 全通过；valid-triplet median `rel_l2=4.72e-3`，
  负对照/valid 分离度为 `316.6×`；120/120 个 valid 输入的 gate/up/product 坐标对齐与
  canonical-down 均 bitwise 相等。
- Stage C：23 个 case × 32 prompt 全部完成，baseline 重复 forward/streams bitwise 相等，
  最终全 MLP SHA 与开始一致。

all-36 结果：

| seed | logits median rel-L2 | top-1 agreement | last-token top-1 | greedy exact match |
|---:|---:|---:|---:|---:|
| 42 | 1.51e-2 | 98.01% | 31/32 | 24/32 |
| 43 | 1.48e-2 | 98.33% | 30/32 | 24/32 |
| 44 | 1.56e-2 | 98.25% | 31/32 | 24/32 |

结论与 Qwen3-4B 后训练版本一致：正确联动 permutation 在数学上成立，BF16 的可见差异
首次出现在被置换层的 `mlp_out`，负对照则产生数量级更大的破坏。Base 模型的 all-36
logits 漂移略低，但短生成 exact match 也略低；这与 Base prompt 上较小的 token margin 相容，
不应把文本 exact match 单独解释为能力变化。

## 运行命令

```bash
cd experiments/ffn_permutation
CUDA_VISIBLE_DEVICES=0 conda run -n qwen3 python probe_synthetic.py \
  --results-dir results_base
CUDA_VISIBLE_DEVICES=0 conda run -n qwen3 python probe_single_mlp.py --resume \
  --model-path ../../../models/Qwen3-4B-Base --results-dir results_base
CUDA_VISIBLE_DEVICES=0 conda run -n qwen3 python probe_full_model.py --resume \
  --model-path ../../../models/Qwen3-4B-Base --results-dir results_base
conda run -n qwen3 python summarize.py --results-dir results_base \
  > results_base/summary.json
```
