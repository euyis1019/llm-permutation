# General Benchmark 协议约定

本文只保留 llm-brewing 仍需继承的 benchmark 行为约定。旧 bundle 的独立执行与调度流程不再是当前
项目接口；主实验统一从 `python -m brewing --config ...` 进入。

## 单一行为来源

每个 normalized benchmark 的 `benchmark_meta.json` 描述原始行为契约：

- prompt builder；
- scorer；
- stop tokens；
- few-shot 数量与示例；
- generation kwargs；
- 数据选择和 provenance。

suite/config 负责选择“评什么”，不应静默覆盖“如何构造 prompt 和评分”。如果当前研究需要把 CoT 协议
改成 direct-answer next-token 协议，应创建带版本的新协议，并在 artifact metadata 中记录，而不是悄悄
修改原 metadata 的含义。

## Next-token 实验约定

进入当前 anchor 实验的样本必须满足：

1. 最终 prompt 和 few-shot 示例可完全复现；
2. ground-truth 答案紧接 prompt 出现在第一个生成位置；
3. 在 anchor tokenizer 下，gold token 的字符串、前导空格规则和 token id 有唯一映射；
4. scoring 使用同一映射，不从完整 generation 反推首 token；
5. sample id、benchmark、domain、subject 和 split 被保留。

MC 任务还应检查：

- 选项标签集合是否一致；
- 标签在目标上下文边界下是否各为一个 token；
- label frequency 是否平衡；
- 是否执行 option shuffle，以及 shuffle 后 GT 是否同步；
- train/eval 是否共享题目、few-shot 示例或近重复样本。

## Brewing adapter 约定

adapter 输出 `brewing.schema.Sample` / `DatasetArtifact`，并至少保留：

```text
source_benchmark
source_sample_id
protocol_version
prompt
ground_truth_answer
ground_truth_token_id
domain / subject
split
```

数据读取不得同时加载 full file 与其 row/subject slices。路径解析必须相对仓库或显式 config，metadata 中的
历史绝对路径仅用于 provenance。

## 训练与评估隔离

- label space 只能由 training split 构造并冻结；
- validation/eval 中的新 token 必须按预先定义的 unknown policy 处理；
- probe 训练样本是否按模型正确性筛选，必须是显式配置并写入 artifact metadata；
- global probe 与 per-domain probe 使用不同 artifact id，不能覆盖；
- few-shot demonstrations 不得来自待评估样本。

## 回归要求

迁移任何 prompt/scorer 逻辑时，至少加入：

- metadata load/serialize 测试；
- prompt golden test；
- tokenizer boundary test；
- scorer golden test；
- split 去重测试；
- CUE-Bench 现有路径不退化的回归测试。

总体设计属于原评测基建；本仓库以 `experiments/ffn_benchmark_eval/configs/` 下的冻结配置为准。
