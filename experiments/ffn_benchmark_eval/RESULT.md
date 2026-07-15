# Qwen3-4B / Qwen3-4B-Base FFN permutation — benchmark 等价性与波动实验结论

> 执行日期：2026-07-11
> 预注册方案：[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)
> 上游实验（logits/activation 层）：[`../ffn_permutation/RESULT.md`](../ffn_permutation/RESULT.md)
> 模型：`Qwen3-4B`（instruct）、`Qwen3-4B-Base`；全程 BF16
> 环境：`vLLM 0.24.0` / `PyTorch 2.11.0+cu130` / `Transformers 5.13.0`，conda `qwen3`
> 复现：原始数据在 [`results/raw/`](results/raw)，聚合在 [`results/*.json`](results)，图在 [`results/figures/`](results/figures)

> **后续勘误阅读提示（2026-07-12）：** 本文保留了最初提交时关于 adjacent-swap 与“位移越大影响越大”的原始判断，作为实验历史。该机制判断已由 §8 的复审勘误推翻：本实验的偶对齐 adjacent-swap 与零效应一致，原始非零分数落在 Instruct 自身运行地板内。阅读 §0、§3 和 §4 的幅度结论时，以 §8 为准。

## 0. 一句话结论

**正确的全 FFN 联动 permutation 在精确实数域不改变函数，在 BF16 部署下对 benchmark 的影响“很小但真实存在、并非纯随机震荡”：多选题（MMLU/C-Eval/CMMLU，单 token 答案）几乎完全不受影响（测试了 20 组随机置换，由 20 个不同随机种子各生成一套置换；其中 0/20 组越过 ±1pp，correctness 一致率 ≥99%）；影响集中在需要长文本生成的 benchmark（GSM8K、HumanEval+、MBPP+），六任务平均分变化约 0.5–0.8pp，且方向随模型固定（instruct 系统性略降 −0.79pp，base 系统性略升 +0.49pp），因此不是对称噪声。影响大小完全由“BF16 `down_proj` 归约顺序被扰动的程度 × 扰动传播的深度”决定：permutation 越靠前的层、位移越大，影响越大；只置换最后一层或只做相邻交换几乎无影响。**

对两个预注册问题的直接回答：

- **Q1（影响多大）**：见 §3。单 benchmark 层面，MC 类实用等价；生成类 GSM8K/HumanEval+/MBPP+ 有约 1–2.3pp 的真实偏移。两模型的六任务平均分都**略微超出**预注册的 ±0.5pp 实用等价区间（instruct 19/20 组、base 11/20 组越界），所以严格判据下**不能**判为六任务整体完全等价，但也远小于“换个模型/训练”级别的差异。
- **Q2（与 permutation 选取方式的关系）**：见 §4。**局部 vs 全局**：主导因素是扰动**传播深度**而非置换层数——只置换第 0 层就已产生全 36 层约 75% 的影响，只置换最后一层几乎无影响。**幅度大小**：大位移（random/reverse）产生完整影响，最小位移（相邻交换 adjacent-swap）影响只有约 1/4，因为相邻交换几乎不改变 `down_proj` GEMM 的浮点归约顺序。

## 1. 方法与有效性（对应 §8 判据）

- **样本**：MMLU/C-Eval/CMMLU 各 500 条按 subject 确定性分层抽样，GSM8K 固定 500 条 ID，HumanEval+（164）/MBPP+（378）全量。选择清单与 SHA 见 [`configs/sample_selection_manifest.json`](configs/sample_selection_manifest.json)。评测协议（prompt builder、scorer、few-shot、stop、EvalPlus tests）直接硬拷贝自 `bench/`，未改动。
- **推理确定性（关键）**：默认 vLLM greedy 在长文本生成上**不是逐 run 可复现**的（连续批处理 + BF16 归约顺序噪声）：同权重重复运行 GSM8K 有 ~15/100 的答案翻转。为消除该混淆，冻结 `VLLM_BATCH_INVARIANT=1` + `enforce_eager=True` + 关闭 prefix caching，并把每个模型族固定在同一张 GPU（两张 RTX 4090 非逐比特一致）。详见 [`configs/frozen_config.json`](configs/frozen_config.json) 与 memory `vllm-determinism-benchmark`。
- **baseline_copy 对照**：不置换、按相同流程重存的 checkpoint，与原始 baseline 全 tensor 逐字节一致，评测结果**逐样本完全相同**（所有 benchmark Δ=0.00pp）——确认 checkpoint 重写与评测系统本身不引入差异。
- **噪声地板（null 分布）**：每个模型族跑 11 个同权重 baseline（原始×2、copy、rep02–09），两两配对得到“同函数重跑”的波动分布，作为判断 permutation 是否只是噪声的基准。
  - `qwen3_4b_base`：**全部 6 个 benchmark 的 correctness 完全确定**（11 个 baseline 两两 max|Δ|=0.00pp，disagreement 0%）。因此 base 上任何 permutation 偏移都是**真实效应，不是推理噪声**。
  - `qwen3_4b`：MC 与两个代码 benchmark 也完全确定（0.00pp）；**唯独 GSM8K 残留** batch-invariant 未能完全消除的 chunked-prefill 噪声（同权重两两 max|Δ|=2.40pp、disagreement 5.97%）。故 instruct 的 GSM8K 数值需对照此地板解读，其余 benchmark 无此顾虑。
- **配置偏离记录**：实验中途另一用户占用 GPU0 约 29GB，原 `gpu_memory_utilization=0.90` 反复 OOM。已验证结果对 `gpu_memory_utilization` 不变（0.40 vs 0.90 在 base GSM8K+MMLU 上 correctness 0 差异），故把后续任务降到 0.28 与其共存，并加入 OOM 自动重试；0.90 与 0.28 的结果可混用。

## 2. 实验规模

| 组 | 内容 | checkpoint 数 |
|---|---|---|
| baseline/null | 每族：原始×2 + copy + rep02–09 | 11 × 2 = 22 |
| 阶段一确认 seed | 每族 all-36 random seed 42/43/44 | 3 × 2 = 6 |
| 阶段二多组随机置换评测 | 每族使用随机种子 1000–1019，各生成一套 all-36 随机置换 | 20 × 2 = 40 |
| 消融（仅 qwen3_4b） | scope: single L0/L17/L35, prefix6/18, all36；magnitude: adjacent-swap / reverse | 8 |
| **合计** | | **76** |

每个 checkpoint 跑全部 6 个 benchmark，保存逐样本文本 / 抽取答案 / correctness，用于配对分析。

## 3. Q1：影响到底有多大

### 3.1 阶段二 20 组随机置换的分布（accuracy delta，pp，vs 各族 baseline）

**Qwen3-4B（instruct）**

| benchmark | mean | std | min | max | 5–95% | disagree | #\|Δ\|>1pp | null 地板 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | +0.05 | 0.26 | −0.40 | +0.40 | [−0.40,+0.40] | 0.77% | 0/20 | 0.00 |
| ceval | +0.20 | 0.18 | −0.20 | +0.40 | [−0.01,+0.40] | 0.38% | 0/20 | 0.00 |
| cmmlu | −0.57 | 0.21 | −0.80 | +0.00 | [−0.80,−0.19] | 0.81% | 0/20 | 0.00 |
| **gsm8k** | **−2.33** | 0.80 | −3.80 | −1.00 | [−3.42,−1.19] | 9.01% | 20/20 | **2.40 (噪声大)** |
| **humaneval+** | **−1.71** | 0.81 | −3.05 | +0.61 | [−3.05,−0.55] | 3.17% | 18/20 | 0.00 |
| mbpp+ | −0.37 | 0.56 | −1.06 | +0.79 | [−1.06,+0.54] | 1.93% | 4/20 | 0.00 |
| **六任务平均分** | **−0.79** | 0.19 | −1.11 | −0.32 | — | — | 19/20 >0.5pp | — |

**Qwen3-4B-Base**（null 地板全部为 0，所有偏移均为真实效应）

| benchmark | mean | std | min | max | 5–95% | disagree | #\|Δ\|>1pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| mmlu | +0.45 | 0.22 | +0.00 | +0.80 | [+0.19,+0.80] | 0.79% | 0/20 |
| ceval | +0.45 | 0.27 | +0.00 | +0.80 | [+0.00,+0.80] | 0.75% | 0/20 |
| cmmlu | +0.18 | 0.22 | −0.20 | +0.60 | [−0.20,+0.60] | 0.58% | 0/20 |
| **gsm8k** | **+0.97** | 0.68 | −1.00 | +2.00 | [+0.14,+1.81] | 5.57% | 12/20 |
| humaneval+ | −0.12 | 0.99 | −1.83 | +2.44 | [−1.83,+1.28] | 2.74% | 6/20 |
| **mbpp+** | **+1.03** | 0.59 | −0.26 | +2.12 | [+0.24,+2.12] | 2.75% | 10/20 |
| **六任务平均分** | **+0.49** | 0.21 | +0.13 | +0.99 | — | — | 11/20 >0.5pp |

图：[`results/figures/stage2_seed_delta_*.png`](results/figures)、[`stage2_macro_hist_*.png`](results/figures)。

### 3.2 结论

1. **多选题几乎完全等价。** MMLU/C-Eval/CMMLU 三项在两个模型上的 20 组随机置换**全部**落在 ±1pp 内（0/20 组越界），correctness 一致率 99%+，text/answer 也高度一致。单 token 答案对 BF16 数值路径不敏感——这一层就是“轻微震荡”，甚至连震荡都很小。
2. **影响集中在长文本生成 benchmark。** GSM8K、HumanEval+、MBPP+ 才是敏感项：单 seed 偏移可达 1–3pp。机制是 permutation 改变的 BF16 数值路径在逐 token 生成中沿 512 步不断放大，在“低 margin 决策点”（推理分叉、代码分支）翻转最终答案。
3. **不是对称噪声，而是有方向的系统性小偏移。** instruct 的 GSM8K 在 20 组随机置换下**全部为负**（−2.33 均值），base 的 GSM8K/MBPP+ 显著偏正。若只是随机震荡，均值应接近 0；这里每个模型都有稳定的符号，说明是真实的数值路径效应，而非纯抖动。方向随模型不同（instruct 略降、base 略升），没有普适方向。
4. **量级定位。** 六任务平均分变化约 0.5–0.8pp。这**超出**预注册 ±0.5pp 实用等价区间（instruct 19/20 组、base 11/20 组越界），因此严格判据下六任务整体存在可测但很小的差异，而非完全等价；但它比 MC 类的噪声地板大一个量级，又比“重训/换模型”小得多。GSM8K（instruct）因自身推理噪声地板高达 2.4pp 而部分被混淆，但 HumanEval+（−1.71pp，null=0）与 base 全部 benchmark（null=0）证明**效应本身是真实的**。

### 3.3 阶段一三确认 seed（配对 bootstrap 95% CI + McNemar）

- instruct 三 seed 的六任务平均分变化为 −0.61 / −0.40 / −0.90 pp（CI 均含 0，如 s42 `[−1.97,+0.75]`）；base 三 seed = +0.41 / +0.64 / +0.69 pp（CI 均含 0）。逐 benchmark McNemar exact p 多数 >0.05（单 seed 500 样本功效不足以判定生成类 1–2pp 偏移显著），只有 base GSM8K、instruct HumanEval+ 接近显著。**结论只能靠汇总 20 组随机置换得到，单个随机种子生成的一组置换不足以定论**——这正是预注册要求评测多组随机置换的原因。详见 [`results/stage1_summary.json`](results/stage1_summary.json)。

## 4. Q2：影响与 permutation 选取方式的关系（消融，qwen3_4b）

对同一 baseline，固定其它条件只改一个轴。六任务平均分变化（pp）与平均 correctness disagreement：

| 轴 | arm | 说明 | macro Δpp | disagreement |
|---|---|---|---:|---:|
| **scope（层位置/数量）** | single L0 | 只置换第 0 层 | −0.61 | 2.70% |
| | single L17 | 只置换中间层 | −0.21 | 2.06% |
| | single L35 | 只置换最后一层 | −0.17 | 1.03% |
| | prefix 6 | 前 6 层 | −0.75 | 2.57% |
| | prefix 18 | 前 18 层 | −0.84 | 2.73% |
| | all 36 | 全部层（=主实验） | −0.83 | 2.67% |
| **magnitude（位移大小）** | adjacent-swap | 全 36 层，仅相邻两两交换（位移 ≈1） | **−0.20** | 1.20% |
| | reverse | 全 36 层，整体倒序（位移 ≈4864） | −0.46 | 2.81% |
| | random | 全 36 层，随机置换（位移 ≈3242，=主实验） | −0.83 | 2.67% |

图：[`results/figures/ablation_scope_magnitude.png`](results/figures)。

### 结论

1. **局部 vs 全局：起决定作用的是“扰动传播的深度”，不是“置换了多少层”。**
   - 只置换**第 0 层**（single L0，−0.61pp）就已达到全 36 层（−0.83pp）约 **75%** 的影响；而只置换**最后一层**（single L35，−0.17pp）几乎无影响。
   - 位置越靠前，影响越大：L0 (−0.61) > L17 (−0.21) > L35 (−0.17)。因为靠前层引入的 BF16 扰动要穿过全部下游层不断被放大；最后一层的扰动只经过 lm_head 一步。
   - 因此“全局 permutation”并不比“把扰动注入到最前面”坏多少——影响随传播深度**快速饱和**。这与上游 logits 实验“越早注入放大越多、all-36 ≈ 单独置换 L0”的结论一致。
2. **幅度大小：真的有关系，且由“归约顺序被打乱的程度”决定。**
   - **最小位移的 adjacent-swap 影响最小**（−0.20pp，仅为 random 的约 1/4、reverse 的约 1/2）。因为相邻交换只调换 `down_proj` 求和中相邻两项——浮点加法对相邻项交换近似不变（上游实验证其 BF16 逐比特不变），几乎不改变归约顺序。
   - 大位移的 random / reverse 彻底打乱 `down_proj` GEMM 的归约结合顺序，产生完整影响。
3. **两轴统一到同一机制：** benchmark 影响 ∝（`down_proj` BF16 归约顺序被打乱的程度）×（该扰动向后传播的深度）。“做局部小置换”影响很小，“做靠前的大位移置换”影响最大——但即便最大也只有六任务平均分约 0.8pp。

## 5. 与上游 logits 实验的一致性

上游（[`../ffn_permutation/RESULT.md`](../ffn_permutation/RESULT.md)）在 logits/activation 层测得：all-36 valid 置换 logits `rel_l2≈2e-2`、top-1 一致率 ~98%、翻转集中在低 margin token、越早注入放大越多。本实验在 benchmark 层完全印证并量化了其下游后果：翻转确实集中在“低 margin 决策”，因此单 token 的 MC 任务几乎不受影响，而长生成任务因逐 token 累积而显现 1–2pp 的真实偏移；scope/magnitude 消融也复现了“传播深度主导 + 相邻交换近似无害”的机制。

## 6. 原始数据索引

| 路径 | 内容 |
|---|---|
| `results/raw/<tag>/<bench>.raw.json` | 每 checkpoint 每 benchmark 的逐样本文本/抽取答案/correctness |
| `results/stage1_summary.json` | 确认 seed + baseline_copy 的配对分析、CI、McNemar、determinism |
| `results/null_distribution.json` | 11 baseline 两两配对的同函数噪声地板 |
| `results/stage2_distribution.json` | 20 组随机置换的分布（mean/std/median/IQR/分位/range、越界计数） |
| `results/ablation_summary.json` | scope × magnitude 消融 |
| `results/figures/` | seed 分布箱线图、macro 直方图、消融图 |
| `configs/frozen_config.json`, `configs/sample_selection_manifest.json` | 冻结配置与样本清单 |
| `model_manifests/*.sha256` | 原始模型文件 SHA-256 |

## 7. 与预注册判据的对照与偏离

- **§8.1 有效性前提（baseline 逐样本完全一致）**：base 通过（correctness 全等，仅 5 条 MBPP+ 文本差异、0 翻转）；instruct 除 GSM8K 外通过，GSM8K 有 batch-invariant 未消尽的残留噪声，已用 null 地板显式量化而非用 avg@k 掩盖（符合 §8.1“先定位非确定性”的要求）。
- **§8.2/8.3 阶段一强等价**：MC 类满足；生成类单 seed CI 因样本量过宽（多数含 0），按 §8.6 记为“证据不足以在单 seed 判定”，改由汇总 20 组随机置换定论。
- **阶段二结论**：如 §3 所述，六任务平均分变化略超 ±0.5pp，判为“存在很小但真实、方向随模型固定的差异”，非纯随机波动。
- **偏离**：(1) 因另一用户占卡，中途从双卡改为单卡 GPU0、`gpu_memory_utilization` 0.90→0.28，已验证结果不变并混用；(2) 计划的 vLLM 吞吐校准简化为直接采用 batch-invariant + eager 的确定性配置（determinism 优先于吞吐）。

## 8. 勘误（2026-07-12，来自 permutation_min_cost 阶段一复审）

**原 §4 / TL;DR 中"adjacent-swap 影响约为 random 的 1/4"的读法不成立，应更正为"adjacent-swap 与零效应完全一致"。**

依据（均出自本实验已有数据 + 下游单层测量）：

1. `mag_adjacent_swap_all36`（instruct）6 个 benchmark 中 5 个的行为不一致率**恰好为 0**、Δ 恰好为 0；唯一非零的 GSM8K（不一致率 7.2%、Δ −1.2pp）完全落在 instruct 自身 null 地板内（11 个同权重 baseline 两两配对：不一致率均值 6.0%、最大 7.8%；Δ q05=−2.4pp）。六任务平均分的 −0.20pp 是把 null 噪声当成了效应。
2. 下游 `experiments/permutation_min_cost` 的单层测量证实：本实验所用 adjacent_swap（偶对齐 (2k,2k+1) 全交换）在 torch BF16 GEMM 与 vLLM batch-invariant kernel 下、L0/L17/L35、prefill/decode 各形状均**逐比特免费**。全 36 层逐比特免费 ⇒ 端到端 logits 逐比特相同 ⇒ benchmark 零效应，与第 1 条一致。
3. 注意这不是"位移小就无害"：奇对齐 (2k+1,2k+2) 的 distance-1 交换在 torch BF16 下漂移直接达到饱和天花板（~2.7e-3，与全局随机同量级）。无害的充分条件是**保持对齐 8-块成员关系**，而非位移小。原 §4"位移越大影响越大"的表述在机制上应更正为"是否跨越 kernel 归约块边界、以及跨越后的传播深度"。

reverse / random 的结论不受影响（其不一致率在 5/6 个 benchmark 上超出 null 地板最大值，效应真实）。详见 `../permutation_min_cost/review/REVIEW_stage1.md`。
