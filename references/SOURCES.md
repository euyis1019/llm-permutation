# 外部代码来源登记

文献调研阶段曾在本地克隆以下公开仓库。为避免把第三方 Git 历史和大文件打包进新仓库，完整克隆已清理；这里保留当时核对的来源与固定 commit，便于按需恢复。

| 名称 | 上游仓库 | 固定 commit |
|---|---|---|
| CoRP | https://github.com/oooranz/CoRP.git | `4cba6bf9102d` |
| DiZO | https://github.com/Skilteee/DiZO.git | `b05e73d2016f` |
| FZOO | https://github.com/DKmiyan/FZOO.git | `d4f8a8b16eb6` |
| HiZOO | https://github.com/Yanjun-Zhao/HiZOO.git | `4f9905582398` |
| HyperscaleES | https://github.com/ESHyperscale/HyperscaleES.git | `b77f7d6f9123` |
| LOZO | https://github.com/optsuite/LOZO.git | `5d1ade5bf418` |
| QES | https://github.com/dibbla/Quantized-Evolution-Strategies.git | `fefc7358decb` |
| QZO | https://github.com/maifoundations/QZO.git | `ac6803ab2ce7` |
| QuZO | https://github.com/lloo099/QuZO.git | `e4627fadc59e` |
| RLR-Optimizer | https://github.com/RTkenny/RLR-Optimizer.git | `296485caaf41` |
| RandOpt | https://github.com/sunrainyg/RandOpt.git | `536df0a308f3` |
| SensZOQ | https://github.com/GarlGuo/SensZOQ.git | `5219dd6480e9` |
| SubZero | https://github.com/zimingyy/SubZero.git | `cf6effdd0aef` |
| ZO-Fine-Tuner | https://github.com/ASTRAL-Group/ZO_Fine_tuner.git | `ef9abff5f74d` |
| ZO2 | https://github.com/liangyuwang/zo2.git | `4bca25c2cd69` |
| ZOO-Prune | https://github.com/AIM-SKKU/ZOO-Prune.git | `c0558b1049de` |
| es-fine-tuning | https://github.com/VsonicV/es-fine-tuning-paper.git | `574a9d134da1` |

## 仓库内第三方评测资产

下表登记当前仓库实际包含的第三方代码和数据。`未保留` 表示整理新仓库时没有找到足以证明精确上游 revision 的记录；这是需要补齐的 provenance 缺口，不应用猜测值替代。

| 资产 | 本地路径 | 上游来源或版本 | 许可证 / 状态 |
|---|---|---|---|
| EvalPlus 参考实现 | `experiments/ffn_benchmark_eval/bench/src/eval/infra/reference_evalplus/` | https://github.com/evalplus/evalplus；固定 commit 未保留 | 上游代码许可证为 Apache-2.0；本地保留 `LICENSE` |
| HumanEval+ 数据 | `experiments/ffn_benchmark_eval/bench/datasets/benchmark/evalplus/HumanEvalPlus-v0.1.10.jsonl` | `v0.1.10`；下载 URL 见同目录 `MANIFEST.yaml` | 再分发条件待按上游 release 核对 |
| MBPP+ 数据 | `experiments/ffn_benchmark_eval/bench/datasets/benchmark/evalplus/MbppPlus-v0.2.0.jsonl` | `v0.2.0`；下载 URL 见同目录 `MANIFEST.yaml` | 再分发条件待按上游 release 核对 |
| HumanEval 基础数据 | `experiments/ffn_benchmark_eval/bench/datasets/benchmark/evalplus/HumanEval.jsonl` | OpenAI HumanEval；精确 revision 未保留 | 再分发条件待核对 |
| sanitized MBPP 基础数据 | `experiments/ffn_benchmark_eval/bench/datasets/benchmark/evalplus/sanitized-mbpp.json` | Google Research MBPP；精确 revision 未保留 | 再分发条件待核对 |

## 标准化 benchmark 副本

`experiments/ffn_benchmark_eval/bench/datasets/benchmark/normalized/` 是从原 `llm-brewing/bench` 工作区硬拷贝的冻结副本。实验计划保留了源机器路径，但没有保留源 checkout commit、完整的拷贝前后 SHA-256 清单或逐数据集许可证映射。

当前副本包含 BBH、C-Eval、CMMLU、CRUXEval、GSM8K、MATH-500、MMLU、MMLU-Pro 和 MMLU-Redux。九项数据的精确上游 revision 与再分发条件仍需逐项核对；在完成核对前，公开发布这部分数据属于明确的待决事项。内部实验复核应使用现有冻结文件和 `experiments/ffn_benchmark_eval/configs/sample_selection_manifest.json`，不要以最新版上游数据替换历史输入。

## 发布前待办

1. 为 EvalPlus vendored snapshot 找回或重建可验证的上游 commit 对应关系；
2. 为九项标准化 benchmark 登记上游仓库、数据版本、许可证和本地转换来源；
3. 核对 HumanEval+、MBPP+ 及基础数据的再分发条款；
4. 若无法确认某项数据允许公开再分发，应从公开远端排除该数据并提供校验后的下载/重建流程；
5. 根目录项目许可证必须由仓库所有者选择，不能由第三方资产的许可证代替。
