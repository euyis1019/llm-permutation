# 实验产物保留策略

新仓库保留能支撑结论核对和轻量分析重跑的文本产物，同时排除可重建的大型二进制文件。

## 保留

- 所有实验脚本、冻结配置、样本选择 manifest 和模型哈希；
- 预注册计划、失败报告、复审记录与最终结论；
- 汇总 JSON、测量 JSONL 和逐题 benchmark 原始记录；
- 已执行的核心 Notebook 及其内嵌图表；
- 生成大型产物所需的 checkpoint 与 logits 脚本。

## 已清理

| 类型 | 原位置 | 原因 |
|---|---|---|
| 三个派生模型检查点 | `experiments/noise_floor/checkpoints/` | 合计约二十三 GB，可由 checkpoint 脚本重建 |
| 逐 prompt logits 数组 | `experiments/noise_floor/results/**/logits/` | 合计约二点三 GB，汇总与补充指标已经落盘 |
| 实验运行日志 | 各实验的 `logs/` | 调度过程产物；最终状态已进入 manifest、报告和验收 JSON |
| Python 与 Notebook 缓存 | 各级缓存目录 | 与实验结论无关，可自动生成 |
| 外部论文代码完整克隆 | 原 `survey_para_disturbance/` | 不应作为本仓库 vendored source；来源和固定 commit 已登记 |

大型产物的路径已加入 `.gitignore`，避免后续复现实验时误提交。

## 重建入口

- 最终轮派生 checkpoint：`experiments/noise_floor/scripts/make_v11_checkpoint.py`
- 最终轮 logits：`experiments/noise_floor/scripts/run_logits_unit.py`
- 高斯扫描：`experiments/noise_floor/scripts/run_part6_units.py`
- Benchmark 全流程：`experiments/ffn_benchmark_eval/scripts/scheduler.py`
- 核心 Notebook：`notebooks/build_core_notebook.py`

原始模型权重通过仓库根目录的 `models` 本地链接提供，不纳入版本控制。

## 当前轻量仓库规模

首次整理时，纳入版本控制候选的文件约 170 MB，主要由 benchmark 逐题原始记录和冻结评测数据构成；没有单文件超过 GitHub 的 50 MB 提示线。该数字只用于帮助维护者理解克隆成本，不是冻结验收值，后续以 `git ls-files` 和实际对象大小为准。

文本 raw 记录虽然体积较大，但它们支撑逐题复核、Notebook 图表重建和 reviewer 分析，因此按本文件策略保留。第三方 benchmark 数据是否适合公开再分发是独立的许可问题，见 [`references/SOURCES.md`](references/SOURCES.md)，不能仅因文件体积较小就默认可公开。
