# FFN Permutation Experiments

本仓库研究一个具体问题：对 Transformer FFN 的中间通道做数学等价的联动置换时，有限精度 GPU 执行会产生多大差异，哪些置换可以逐比特保持不变，这些差异又会怎样传到生成与 benchmark。

现有 Python 探针已经支持外部用户在自己的受支持 GPU 上复现核心实验，并生成独立的 JSON、JSONL 和环境记录；完整 benchmark 流程的跨机器兼容仍在整理中。后续开发项与当前状态见 [dev_list.md](dev_list.md)。

## 从这里开始

- **核心可执行导览**：[ffn_permutation_core.ipynb](notebooks/ffn_permutation_core.ipynb)
- **整体叙述报告**：[overall_report.md](docs/reports/overall_report.md)
- **逐实验索引**：[experiment_index.md](docs/reports/experiment_index.md)
- **静态可视化首页**：[docs/index.html](docs/index.html)

Notebook 已执行并保留输出。它直接读取仓库内的机器可读汇总和逐题结果，无需模型权重或 GPU，即可重建核心图表并检查关键结论。

## 主要结论

1. 正确置换必须同步作用于 `gate_proj`、`up_proj` 和 `down_proj`；错误配对与正确联动之间有明确的数量级分离。
2. 有限精度差异首次出现在 `down_proj` 的并行归约路径。
3. 在当前 RTX 4090、BF16 与冻结 vLLM 栈中，对齐八通道块内置换可从单层一直保持到 benchmark 逐题层面的零差。
4. 跨块置换会产生 backend 相关的非零漂移；影响主要出现在长生成任务和低 margin 决策点。
5. 当前结果不支持通过在评测集上挑选置换随机种子获得可迁移的能力提升。

这些结论的严格适用范围与例外见 [current_findings.md](docs/reports/current_findings.md)。

## 目录结构

```text
notebooks/        核心可执行导览及其构建脚本
experiments/      按问题拆分的代码、预注册计划、汇总结果和原始文本结果
docs/reports/     面向读者的整体报告、实验索引与阶段认识
docs/plans/       跨实验设计文档
docs/             静态可视化页面
references/       文献综述、外部代码来源登记和轻量方法抽象
scripts/          独立的最小推理入口
```

## 重新运行核心 Notebook

先安装 [requirements-notebook.txt](requirements-notebook.txt) 中的轻量依赖，然后在仓库根目录运行：

```bash
python notebooks/build_core_notebook.py
jupyter nbconvert \
  --to notebook \
  --execute notebooks/ffn_permutation_core.ipynb \
  --output ffn_permutation_core.ipynb
```

这一步只读取已有结果。完整 GPU 实验的复现入口、冻结配置和硬件边界分别记录在各 `experiments/*/README.md`、计划文档与结果报告中。

## 数据与大文件策略

仓库保留可审计的 JSON、JSONL、逐题 benchmark 记录、配置、manifest 与分析脚本。派生模型检查点、逐 prompt 二进制 logits、缓存和运行日志不进入版本控制；它们体积大且可由保留脚本重建。具体清理范围与重建入口见 [ARTIFACTS.md](ARTIFACTS.md)。

仓库已经按面向复核与复现的新布局整理完毕；首次发布前仍需确认远端可见性、根许可证与第三方 benchmark 的再分发条件。
