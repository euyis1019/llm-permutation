# noise_floor 简短执行报告

> 执行日期：2026-07-13  
> 唯一执行依据：`EXPERIMENT_PLAN.md`  
> 状态：完成；三个硬判据全部通过，无需 `FAILURE_REPORT_noise_floor.md`。

## 验收结论

| 判据 | 结果 | 关键观测 |
|---|---|---|
| S0-1（硬） | PASS | 两个独立 Base 进程 32/32 last-token logits 原始字节一致，max\|Δ\|=0 |
| S1-1（硬） | PASS | F9-K100-all36 与 identity 32/32 逐比特一致 |
| S1-2（软） | PASS | F10/F7 均 0/32 逐比特一致；rel_l2 中位数比 0.960544 |
| S1-3（硬） | PASS | F9 六 benchmark 逐题 correctness 差异全为 0，六任务平均分变化为 0 |
| S1-4（软） | PASS | F10/F3 平均逐题不一致率 1.8818%/2.4614%，比值 0.764520 |
| S2-1（软） | FAIL | 8 锚点 logits 漂移与既有不一致率 Spearman ρ=0.595238（阈值 0.9） |
| P6-1（预测） | FAIL | σ=1e-6 已 0/96 逐比特一致，未出现“更小档全零”的平台前区 |
| P6-2（预测） | FAIL | σ* 在最低网格点即已达到 F7 线，记作 σ*≤1e-6，不满足 ≥1e-4 |

## 主要测量

- Part 1a 的 logits rel_l2 中位数：F9=0，F10=0.00948927，F7=0.00987905。
- Part 1b 的 F9 correctness 全零差；响应字节除 GSM8K 497/500 一致外，其余五项 100% 一致。硬判据按预注册 correctness 层判定。
- Part 2 只有 prefix-6 在 32 prompt 中出现 1 次 top-1 翻转，其余锚点为 0；相关性未达到预注册阈值。
- Part 6 权重量化端点：σ=1e-6 时 1.3118% 参数实际改变、权重 rel_l2=8.018×10⁻⁶；σ=1e-2 时为 98.9562%、0.388647。
- Part 6 logits 曲线在 σ=1e-6 的中位 rel_l2 已为 0.0099435，是 F7 参考线的 1.0065 倍；σ=1e-2 时升至 1.08241。
- σ* 不在 `[1e-4,1e-2]`，所以 Part 6c 按条件跳过，没有替换档位或追加 seed。

## 资源与审计

- 全程仅使用 GPU 0；每次引擎启动前实时检查显存，正式 GPU 单元均采用 `gpu_memory_utilization=0.28` 正常档。
- 无 OOM、无降配、无 600 秒退避；同一时刻最多一个测量进程。Part 6 临时 checkpoint 用完即删，最终不存在临时权重。
- Part 1b 的三个正式 checkpoint 按计划保留；Part 2 的旧锚点逐个再生、测量并恢复为仅 manifest 状态。
- Part 3/4/5/7 未执行；NLL 已随 Part 0/1a/2/6b 记录，留给 reviewer 的 Part 7 分析。

机器可读总验收见 `results/acceptance_noise_floor.json`，逐文件 SHA-256 与环境见 `results/manifest.json`。
