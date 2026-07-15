# 最终轮执行计划:数值噪声下限与扰动谱(noise_floor)

> 日期:2026-07-12
> 设计依据:`docs/plans/final_round_design.md`(动机与结果解读都在那里,本文件只写执行)
> 背景结论:`docs/reports/current_findings.md`
> 本计划是预注册文档:§7 的判据与预测写下后不许改。执行者(codex)负责 Part 0/1/2/6 的 GPU 测量与数据落盘;Part 3/4/5/7 的分析由 reviewer 在已有数据上完成,不在本计划范围内。

---

## 1. 资源纪律(最高优先级,先读这节)

这台机器不是我们独占的,算力现在很紧张。所有 GPU 工作必须实时根据机器当前状态申请资源、调整参数,规则如下:

1. 只用 GPU 0(`CUDA_VISIBLE_DEVICES=0`),任何情况下不碰 GPU 1。
2. 每次启动引擎(或任何要上 GPU 的进程)之前,先跑 `nvidia-smi` 查 GPU 0 的空闲显存,按下表决定:

| GPU 0 空闲显存 | 动作 |
|---|---|
| ≥ 15 GiB | 正常启动,`gpu_memory_utilization=0.28` |
| 10 到 15 GiB | 降配启动:util 取 (空闲 − 2 GiB) / 总显存,但不低于 0.18;同时把 `max_model_len` 降到 2048、`max_num_seqs` 降到 8 |
| < 10 GiB | 不启动。等 600 秒重新查,如此循环;累计等待超过 4 小时,把当前进度写进 `PROGRESS.md` 后正常退出(后续可断点续跑) |

3. 同一时刻最多一个测量进程占 GPU。两次引擎加载之间,确认上一个进程已退出、显存已释放,再查一次 nvidia-smi。
4. 长任务(Part 1b 的 checkpoint × benchmark)以单元为粒度:每个单元开始前重新执行第 2 条检查,不满足就在单元边界等待;结果逐单元落盘,断点续跑时跳过已完成单元(这不算重跑正式测量)。
5. 运行中如果因为别人进程显存上升而 OOM:不算判据失败,退避(等待后按第 2 条重新申请)并在 `DECISIONS.md` 记一笔;同一单元的重试沿用完全相同的科学参数。
6. `gpu_memory_utilization`、`max_num_seqs`、`max_model_len` 是资源参数,可按上表调整并记录;prompt 集、seed、σ 网格、模型、判据是科学参数,任何情况下不许动。
7. 磁盘:当前 /nvme0 剩约 1.5 TB。Part 1b 需要落盘 3 个 checkpoint(各约 8 GB);Part 6 若采用临时 checkpoint 方案,每个用完立刻删除,任何时刻磁盘上最多存在一个临时 checkpoint。

## 2. 冻结的公共要素

- python:`/nvme0/if/anaconda3/envs/qwen3/bin/python`(anaconda3,不是 miniconda3)
- 模型:Base = `/nvme0/if/models/Qwen3-4B-Base`;Instruct = `/nvme0/if/models/Qwen3-4B`(仅 Part 2 用 Instruct,其余都用 Base)
- 确定性配置(所有引擎运行统一):`VLLM_BATCH_INVARIANT=1`,`enforce_eager=True`,关闭 prefix caching,dtype bfloat16,temperature=0,`TOKENIZERS_PARALLELISM=false`
- prompt 集:`experiments/ffn_permutation/prompts.json` 的全部 32 条,顺序与 id 不变
- logits 记录:每条 prompt 取最后一个 token 的完整 logits,原始 dtype 逐字节保存(逐比特比较就是比较这些字节);另存 float32 副本供算 rel_l2
- NLL 记录:Part 0/1a/2/6 的每次引擎运行,顺带用 prompt_logprobs 记录每条 prompt 的平均 NLL(Part 7 分析用,不设判据)
- 置换族定义沿用 `experiments/permutation_min_cost/scripts/perm_families.py` 与 AMENDMENT_v1.1 的口径;M=9728
- 每步产物带 SHA-256,汇总进 `results/manifest.json`;实现级选择一律记 `DECISIONS.md`

## 3. Part 0:重复运行基线(先做,是后面逐比特判据的前提)

- 操作:Base 模型,不做任何置换,同一确定性配置,开两个独立进程(先后运行即可,不要求同时)各跑一遍 32 条 prompt,各自保存最后 token 的 logits。
- 比较:32 条逐字节比对。
- 产出:`results/part0_run_a/`、`results/part0_run_b/`、`results/part0_compare.json`(每条 prompt 是否逐比特一致、不一致时的 max|Δ| 与 rel_l2)。
- 预计 GPU 时间:两次加载,约 0.5 小时。

## 4. Part 1:块内置换从单层推到整个 benchmark

### 4a. 引擎 logits 级

4 个变体,全部在内存中对权重施加置换(不落盘),各跑同样 32 条 prompt:

1. identity(可直接复用 Part 0 的 run_a 结果,记录清楚即可);
2. F9-K100-all36:全部 36 层做全量块内重排,第 L 层 seed = 401+L;
3. F10-K100-all36:全部 36 层做全部奇对齐对换(确定性,4863 对,端点 0 与 9727 不动,与 v1.1 的 DECISIONS 口径一致);
4. F7-all36:全部 36 层全局随机置换,第 L 层 seed = 402+L。

产出:`results/part1a_logits.jsonl`,每条记录含变体、prompt id、是否与 identity 逐比特一致、rel_l2、top-1 是否翻转、平均 NLL。

### 4b. benchmark 级(仅当 S1-1 通过才执行)

3 个 checkpoint 落盘(用 ffn_benchmark_eval 的 make_checkpoint 流程,Base 模型):

1. F9-K100-all36(预测:与 baseline 逐题零差);
2. F10-K100-all36(预测:与随机对照同量级);
3. F3-K30-all36,seed=301(随机跨块对照)。

每个 checkpoint 跑全部 6 个 benchmark,协议、样本清单、scorer 与 `ffn_benchmark_eval` 完全一致;与已有 `qwen3_4b_base__baseline_original_run1` 逐题比对。

产出:`results/part1b/` 下按 checkpoint × benchmark 落盘 raw 记录,汇总 `results/part1b_compare.json`(逐题 correctness 一致数、response 逐字节一致数、不一致样本清单)。

说明:已知 baseline 自身两次运行在 MBPP+ 上有 5 条文本差异(correctness 不变)。因此硬判据定在 correctness 层;response 层的字节差异单独记录,预期不超过该量级。

预计 GPU 时间:4a 约 0.7 小时;4b 约 6 小时(逐单元续跑)。

## 5. Part 2:锚点的 logits 漂移(Instruct 模型)

- 对象:`experiments/ffn_benchmark_eval/checkpoints/` 下 8 个消融 checkpoint(scope_single0 / single17 / single35 / prefix6 / prefix18 / all36_random_s7 / mag_adjswap_all36 / mag_reverse_all36),直接从盘上加载。
- 操作:另加一次 Instruct 原始模型作为基准,共 9 次引擎加载,各跑 32 条 prompt,记录最后 token logits。
- 产出:`results/part2_anchor_drift.jsonl`,每个锚点一条:rel_l2(对 Instruct 基准)、逐 prompt top-1 翻转数、平均 NLL。已有的逐题不一致率从 `ffn_benchmark_eval/results/` 读取并并入该文件,算 Spearman ρ 写进 `results/part2_summary.json`。
- 预计 GPU 时间:约 1.5 小时。

## 6. Part 6:σ 扫描(高斯扰动对比 permutation 噪声)

### 6a. 权重级统计(CPU,零 GPU,可在 GPU 等待期间做)

对 σ ∈ {1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2},在全模型(Base)所有参数上统计:按 bf16 原位相加后,实际被改变的参数比例、实现的权重 rel_l2,分层与全模型汇总。产出 `results/part6_weight_quant.json`。

### 6b. logits 级(GPU,本 part 核心)

- 配置:上述 10 档 σ × 3 个噪声 seed(seed = 1000 + 10×σ档序号 + rep,rep ∈ {0,1,2}),共 30 个配置。
- 加噪方式与 RandOpt 一致:单个 torch.Generator 以该配置 seed 初始化,按 named_parameters 顺序对全部参数逐个生成 randn×σ,按参数原 dtype(bf16)原位相加。加噪在内存中完成;若实现上必须经过临时 checkpoint,按 §1 第 7 条处理。覆盖的参数范围(是否含 embedding、lm_head、norm)由执行者定,但 30 个配置必须完全一致,并记入 DECISIONS.md。
- 每个配置跑 32 条 prompt,记录:是否与 identity 逐比特一致、rel_l2、逐 prompt top-1 翻转、平均 NLL。
- 参考线:Part 1a 中 F7 与 F10 的 rel_l2(同模型、同 prompt、同栈,内部一致)。
- 产出:`results/part6_sigma_sweep.jsonl` 与曲线汇总 `results/part6_summary.json`(每档 σ 的漂移中位数、与 F7 中位数之比、σ* 的插值估计)。

### 6c. 行为级(条件臂)

仅当 6b 测出的 σ* 落在 [1e-4, 1e-2] 区间内才执行:取两档 σ(最低的产生非零漂移的档,和最接近 σ* 的档)各 5 个噪声 seed(seed = 2000+rep),Base 模型只跑 GSM8K(500 题,冻结协议),逐题落盘。产出 `results/part6c/`。σ* 不在区间内则跳过并在 PROGRESS.md 说明。

预计 GPU 时间:6b 约 5 小时(30 次加载,逐单元续跑);6c 约 2.5 小时(条件执行)。

## 7. 判据与预测(预注册,写下后不许改)

| 编号 | 性质 | 内容 |
|---|---|---|
| S0-1 | 硬 | Part 0 两次独立运行 32/32 逐比特一致 |
| S1-1 | 硬 | Part 1a 中 F9 与 identity 32/32 逐比特一致 |
| S1-2 | 软 | F10 与 F7 均非逐比特,rel_l2 中位数之比在 [1/5, 5] |
| S1-3 | 硬 | Part 1b 中 F9 臂 6 个 benchmark 逐题 correctness 零差 |
| S1-4 | 软 | F10 臂逐题不一致率与 F3 对照之比在 [1/5, 5] |
| S2-1 | 软 | 8 锚点 logits 漂移与已有逐题不一致率 Spearman ρ ≥ 0.9 |
| P6-1 | 预测 | 小 σ 端存在平台:某段 σ 区间内漂移不随 σ 下降,高度与 F7 中位数之比在 [1/3, 3];更小的 σ 档全部逐比特一致 |
| P6-2 | 预测 | σ* ≥ 1e-4,即 RandOpt 网格至少最低一档落在噪声等效区 |

停机规则:

- S0-1 失败:停 Part 1(其逐比特判据失去参照),写 `FAILURE_REPORT_noise_floor.md`;Part 2 和 Part 6 照常执行(它们基于 rel_l2,不依赖逐比特),但结论加注。
- S1-1 失败:跳过 Part 1b,写失败报告;Part 2、Part 6 照常。
- S1-3 失败:如实报告,不影响其他 part。
- 软判据与预测(S1-2/S1-4/S2-1/P6-1/P6-2)不触发停机,只如实记录成立与否。
- 任何硬判据失败后:不调整阈值、不换 seed、不重跑正式测量,立即写报告等待人工审阅。

执行顺序:Part 0 先行;然后 1a;S1-1 通过则 1b;之后 2;最后 6b(6a 可随时在 CPU 上做;6c 条件执行)。GPU 等待期间可做 6a 和其他 CPU 侧工作。

## 8. 产物清单

| 文件 | 内容 |
|---|---|
| `results/part0_*` | 重复运行基线的原始 logits 与比对 |
| `results/part1a_logits.jsonl` | 4 变体 × 32 prompt 的逐条记录 |
| `results/part1b/`、`part1b_compare.json` | 3 checkpoint × 6 benchmark 逐题记录与比对 |
| `results/part2_anchor_drift.jsonl`、`part2_summary.json` | 8 锚点漂移与相关系数 |
| `results/part6_weight_quant.json` | 权重级量化统计 |
| `results/part6_sigma_sweep.jsonl`、`part6_summary.json` | 30 配置漂移记录与 σ* 估计 |
| `results/part6c/` | 条件臂逐题记录(若执行) |
| `results/acceptance_noise_floor.json` | S0-1 到 P6-2 的机器可读验收 |
| `results/manifest.json` | 全部产物的 SHA-256、环境、耗时 |
| `DECISIONS.md` | 实现级选择记录(含每次资源降配与退避) |
| `PROGRESS.md` | 断点续跑状态(如发生等待退出) |

## 9. 禁止事项

1. 不改任何科学参数:prompt 集、seed 表、σ 网格、置换族定义、模型、判据阈值。
2. 正式测量不重跑;资源退避后的重试与断点续跑不算重跑,但必须沿用相同科学参数并记录。
3. 不用 GPU 1,不超过 §1 的显存规则,不在 GPU 忙时抢占。
4. 硬判据失败后立即停对应下游,写失败报告,等待人工审阅;不自行"修复"后继续。
5. 临时 checkpoint 用完即删;Part 1b 的 3 个正式 checkpoint 保留到人工审阅后再定去留,不自行删除。
