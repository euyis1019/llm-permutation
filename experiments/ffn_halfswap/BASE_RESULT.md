# Qwen3-4B-Base half-swap 实验结果

> 执行日期：2026-07-11  
> 模型：`models/Qwen3-4B-Base`，BF16  
> 原始数据：[`results_base/`](results_base/)

使用 Base 专属的 32 条 tokenized prompt，对全部 36 层执行前后半块联动交换。两种条件下
每条 prompt 的 2 次 forward 与 8 次 greedy generation 均完全确定；置换 inverse 后权重
逐 tensor 恢复，最终全 MLP SHA 一致。

| 指标 | Qwen3-4B-Base | 原 Qwen3-4B |
|---|---:|---:|
| logits median rel-L2 | 1.57e-2 | 2.14e-2 |
| 全 token top-1 agreement | 98.01% | 98.25% |
| last-token top-1 相同 | 30/32 | 32/32 |
| 64-token greedy exact match | 23/32 | 28/32 |

Base 模型仍呈现“logits 高度接近，但低 margin 位置可能使生成轨迹分叉”的相同机制。

```bash
cd experiments/ffn_halfswap
CUDA_VISIBLE_DEVICES=1 conda run -n qwen3 python probe_halfswap.py \
  --model-path ../../../models/Qwen3-4B-Base \
  --results-dir results_base \
  --tokenized ../ffn_permutation/results_base/tokenized.json
```
