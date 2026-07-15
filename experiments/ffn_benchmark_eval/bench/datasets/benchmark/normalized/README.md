# 标准化 benchmark 冻结数据

本目录保存从历史 benchmark 工作区硬拷贝并冻结的标准化 JSONL 与协议元数据。它是当前仓库内的只读实验输入，不是可重新下载最新版数据的入口。

## 当前内容

| 目录 | 任务形式 | 当前仓库中的用途 |
|---|---|---|
| `mmlu/` | 多选 | 主 benchmark；含冻结 500 题选择 |
| `ceval/` | 多选 | 主 benchmark；含冻结 500 题选择 |
| `cmmlu/` | 多选 | 主 benchmark；含冻结 500 题选择 |
| `gsm8k/` | 生成 | 主 benchmark；含冻结 500 题选择和 smoke slices |
| `mmlu_pro/` | 推理/多选 | activation 探针输入与协议复核 |
| `mmlu_redux/` | 多选 | activation 探针输入与协议复核 |
| `bbh/` | 推理生成 | activation 探针输入与协议复核 |
| `math500/` | 数学生成 | activation 探针输入与协议复核 |
| `cruxeval/` | 代码推理 | activation 探针输入与协议复核 |

每个目录至少包含主 JSONL 与 `benchmark_meta.json`。部分主实验目录另有 `selected500.jsonl`；这些选择由 `experiments/ffn_benchmark_eval/configs/sample_selection_manifest.json` 固定并校验。切片文件不是独立数据来源，和 full set 同时读取时必须按 `sample_id` 去重。

## 只读与证据边界

- 不要手工格式化、排序或改写 JSONL 与 metadata；
- 需要重新选择样本时，应写入新的结果目录并保留旧 manifest；
- metadata 中的绝对路径是原工作区 provenance，不代表当前运行时仍应访问该路径；
- 当前实验运行时由 `experiments/ffn_benchmark_eval/scripts/common.py` 把协议绑定到本目录的冻结选择。

## 来源缺口

冻结计划记录了这些文件来自原 `/nvme0/if/llm-brewing/bench` 工作区，但源 checkout 的固定 commit、完整拷贝哈希清单以及九项数据逐项许可证没有随副本保留下来。历史转换脚本仍位于 `bench/src/scripts/suite/prepare_bench_data.py`，但仅凭该脚本不能恢复当时的精确上游版本。

因此，本目录当前满足实验复核所需的内容冻结，不等于已完成公开再分发的来源与许可审查。已知信息和待补项见仓库根目录的 [`references/SOURCES.md`](../../../../../../references/SOURCES.md)。

## 轻量检查

```bash
python -m json.tool mmlu/benchmark_meta.json >/dev/null
python -m json.tool ../../../../configs/sample_selection_manifest.json >/dev/null
```

全仓 JSON、JSONL 与选择哈希的检查结果应在每次发布交付中单独报告。
