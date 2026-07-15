# Benchmark source bundle

`bench/` 是 `ffn_benchmark_eval` 使用的冻结评测资产副本，保存标准化样本、prompt/scorer 协议和代码任务参考实现。当前实验不依赖原机器上的 `llm-brewing` 工作区；运行时从本目录读取协议和数据。

## 当前实验实际使用的内容

- `datasets/benchmark/normalized/`：标准化 benchmark 样本与协议元数据；
- `src/eval/benchmark/`：prompt 构建、答案抽取和评分逻辑；
- `src/eval/infra/reference_evalplus/`：代码任务使用的 EvalPlus 参考实现；
- `datasets/benchmark/evalplus/`：冻结的 HumanEval+、MBPP+ 及基础数据；
- `configs/eval_suites/benchmark/` 与 `docs/conventions/`：历史 suite 配置和行为约定。

主实验通过 `../scripts/common.py` 导入协议模块，通过 `../scripts/run_worker.py` 执行六项 benchmark。完整入口见 [`../README.md`](../README.md)。Activation 量级实验也可通过显式 `--bench-dir` 参数读取这里的九项标准化数据。

## 目录边界

```text
bench/
├── datasets/benchmark/
│   ├── normalized/                 # 九项标准化 benchmark
│   └── evalplus/                   # HumanEval+ / MBPP+ 冻结数据
├── src/eval/benchmark/             # 当前使用的协议与评分逻辑
├── src/eval/infra/reference_evalplus/
│                                      # 外部 EvalPlus 参考源，保留上游许可证
├── configs/eval_suites/benchmark/  # 历史 suite 配置
└── docs/conventions/               # 评测行为约定
```

`src/eval/` 还保留了硬拷贝时带入的服务化执行、提交器和模型兼容基础设施。它们不是当前 Qwen3 permutation 实验的入口，其中部分脚本仍含旧机器路径或其他模型的兼容分支；不要仅凭文件名直接启动。后续若继续清理，应先确认没有被冻结配置、结果复算或代码任务调用，再单独提交结构变更。

`src/eval/infra/reference_evalplus/` 是第三方参考实现边界。不要在其中混入本仓库逻辑，也不要做无关格式化。

## 标准化数据

当前目录包含 BBH、C-Eval、CMMLU、CRUXEval、GSM8K、MATH-500、MMLU、MMLU-Pro 和 MMLU-Redux。主 benchmark 实验使用 MMLU、C-Eval、CMMLU、GSM8K 的冻结 500 题选择，以及 HumanEval+、MBPP+ 全量；其余数据用于历史探针或协议复核。

切片文件不是独立数据来源。消费 full set 与切片时必须按 sample id 去重。字段与冻结状态见 [`datasets/benchmark/normalized/README.md`](datasets/benchmark/normalized/README.md)。

## 来源与发布限制

冻结实验计划记录了本目录从原 `llm-brewing/bench` 工作区硬拷贝而来，但当时没有把源 checkout 的 commit、完整文件哈希清单和九项数据的许可证映射一起保存。本仓库现在只能证明当前副本的内容，不能追溯证明它与某个上游 revision 完全一致。

已知来源、已知版本和缺口统一登记在 [`../../../references/SOURCES.md`](../../../references/SOURCES.md)。在补齐标准化数据的上游版本与再分发条件之前，不应把这个 bundle 视为已经完成公开发布合规审查。

## 轻量自检

以下检查不加载模型：

```bash
python -m json.tool datasets/benchmark/normalized/mmlu/benchmark_meta.json >/dev/null
python -m json.tool ../configs/sample_selection_manifest.json >/dev/null
```

完整 GPU 评测需要冻结的 vLLM、CUDA、模型和设备配置；应从 `experiments/ffn_benchmark_eval/scripts/` 的正式入口运行，而不是从本目录的历史提交脚本启动。
