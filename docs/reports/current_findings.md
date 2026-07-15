# FFN Permutation：当前结论与未决问题

> 更新日期：2026-07-13  
> 适用范围：Qwen3-4B / Qwen3-4B-Base，BF16，RTX 4090；主要部署栈为 PyTorch 2.11 + CUDA 13.0 与 vLLM 0.24 batch-invariant kernel。  
> 目的：用尽量少的背景知识，说明目前已经确认了什么、证据边界在哪里，以及下一步真正需要回答什么。

## 一句话结论

FFN 联动置换在数学上严格等价；实际出现的差异来自 GPU 浮点矩阵乘的归约顺序。当前 vLLM 栈中，只在对齐的 8-neuron 小块内部重排可以做到逐比特不变；一旦跨出这个边界，就失去零漂移保证，实质性跨块通常很快进入同一数量级的数值漂移。漂移会传播到 logits，并可能在低 margin 的 token 决策处改变生成路径。已有 20 组随机置换的 benchmark 结果显示：Instruct 模型整体略降，Base 模型整体略升；换半数据验证又显示，从候选池里挑出的额外优势基本不能迁移。候选池相对原始 baseline 的方向仍然存在，但其机制和跨模型可泛化性尚未解决。

## 1. 必要的 GPU / kernel 背景

可以把一次模型计算理解成以下链条：

```text
模型公式
  → 数值格式（这里是 BF16）
  → 框架与库（PyTorch / vLLM / CUDA）
  → 被实际选择的矩阵乘 kernel
  → GPU 硬件执行
```

- **Kernel** 是 GPU 真正执行的一段计算程序。同一个矩阵乘公式可以因输入形状、框架版本、CUDA 库和 GPU 架构不同而选择不同 kernel。
- **GEMM** 是矩阵乘矩阵，**GEMV** 是矩阵乘向量。两者可能采用不同的并行归约方式，因此逐比特行为不一定相同。
- **CUDA 驱动版本不是唯一变量。** 更准确的说法是：有限精度现象取决于整套执行栈，包括 GPU 架构、dtype、矩阵乘 kernel、框架/CUDA 库及其版本；驱动可能影响兼容性或 kernel 选择，但不能把所有差异简单归因于驱动。
- 数学上的置换等价性与 GPU 无关；**受执行栈影响的是有限精度误差的边界和大小**。因此本文中的“块宽 8”是当前实测栈的属性，不是普适常数。

## 2. 数学上必须是三矩阵联动置换

Qwen3 FFN 的核心形式为：

```text
h = silu(W_gate x) ⊙ (W_up x)
y = W_down h
```

要保持函数不变，必须同时：

1. 按同一个 permutation 重排 `W_gate` 和 `W_up` 的行；
2. 按匹配关系重排 `W_down` 的列。

这只是给中间 neurons 改了编号，没有改变每个激活与 `W_down` 权重列之间的配对关系。**单独置换 gate、up 或 down 中的任何一个矩阵都会真正改变函数。**

当前实验确认的是：三矩阵正确联动后，`gate`、`up`、SiLU 和逐元素乘只产生坐标重排；恢复坐标后均逐比特一致。数值漂移首次且唯一出现在 `down_proj` 矩阵乘中，因为它需要沿 intermediate-neuron 维度做求和归约。详细隔离证据见 [FFN permutation 原始实验结论](../../experiments/ffn_permutation/RESULT.md) 和 [漂移来源分析](../../experiments/ffn_permutation/DRIFT_ORIGIN.md)。

## 3. 当前栈中的“免费置换”：保持对齐 8-块

在当前 RTX 4090 + BF16 kernel 上，实验识别出一个有效的 8-neuron 对齐边界：

```text
[0..7] [8..15] [16..23] ...
```

若每个 neuron 始终留在原来的对齐 8-块中，即满足：

```text
floor(i / 8) = floor(permutation(i) / 8)
```

那么块内可以任意重排。在已验证的适用域内，这类 permutation 的输出逐比特不变，不只是“误差很小”。

正式确认结果：

| backend × 形状 | 块内置换结果 |
|---|---:|
| PyTorch BF16 GEMM（T≥2） | 126/126 逐比特一致；结合前轮探针共 504/504 |
| vLLM batch-invariant prefill | 126/126 逐比特一致 |
| vLLM batch-invariant decode（T=1） | 126/126 逐比特一致 |
| PyTorch BF16 GEMV（T=1） | 不适用：存在 ≤6×10⁻⁵ 的偶发微漂移 |

因此，面向当前 benchmark 使用的 vLLM 栈，**只在各自 8-块内部做 permutation，是目前最强且可验证的零代价方案**。完整复审见 [Stage 1b 复审纪要](../../experiments/permutation_min_cost/review/REVIEW_stage1b.md)，原始验收见 [`acceptance_v11.json`](../../experiments/permutation_min_cost/results/acceptance_v11.json)。

## 4. 为什么跨块后通常很快进入同一漂移量级

`down_proj` 的一个输出，本质上是许多乘积项之和。GPU 不会严格从左到右相加，而会并行生成若干部分和，再继续合并这些部分和。浮点加法不满足结合律：

```text
(a + b) + c  ≠  a + (b + c)
```

块内重排仍让 kernel 的各个归约组看到同一批乘积项，因此可以保持相同的部分和。跨块后，乘积项进入了不同的归约组，部分和及其合并顺序发生变化。结果会在最后的 BF16 舍入处使一部分输出跨过相邻量化格点，通常表现为最低有效位发生变化。

这里的 **ULP** 可以直观理解为“当前数值附近两个相邻 BF16 可表示数之间的一个刻度”。BF16 不能表示连续的所有实数；例如在数值约为 10 时，相邻刻度可以是 `10.0000` 和 `10.0625`，舍入边界在两者中间。两条归约路径即使在舍入前只得到 `10.03124` 和 `10.03126`，最终也可能分别保存为上述两个相邻刻度：舍入前只差 `0.00002`，保存后却差 **1 ULP**。若两者没有分处边界两侧，则会舍入成同一个 BF16 数，差异为 0。

因此，当前正确联动置换产生的微小归约扰动，在输出上主要表现为“某个元素不变”或“某个元素跳到相邻 BF16 数”。它通常不足以让单个元素连续跨越很多刻度；当跳 1 ULP 的输出比例很快稳定后，整个张量的 `rel_l2` 也就形成平台。这里的“约一个 ULP”是本实验适用域内的实测现象，不是所有浮点误差都不会超过 1 ULP 的普适定理。

这解释了一个反直觉结果：在 PyTorch BF16 GEMM 中，一旦跨块扰动足以触发这种舍入翻转，漂移大小主要受 **BF16 输出格距（约一个 ULP）** 限制，而不再与移动了多少 neurons 成比例。实测散置 5%、30%、50%、全局随机和反转的单层 `rel_l2` 都约为 `2.4×10⁻³–3.2×10⁻³`。换得更少，并不稳定地换来更小的误差。

需要保留两个边界：

- **任何跨块都会失去“逐比特为零”的保证，但不是每一次跨块都必然立即到达完全相同的平台。** 对齐 16–32 窗附近存在狭窄、数据相关的过渡区。
- 平台高度也是 kernel 属性。vLLM batch-invariant 的非零漂移是约 `2×10⁻⁵–2×10⁻⁴` 的弥散小带，并非 PyTorch GEMM 的紧平台。

因此，工程上最稳妥的表达是：**要获得严格保证，就留在经过验证的 8-块内部；一旦跨块，只能接受 backend 相关的非零漂移，不能依赖“少换一点就一定少漂一点”。** 详细数据和认知修正见 [实验总览 §6.1](../../experiments/permutation_min_cost/OVERVIEW.md#61-重要认知修正实质跨块近似开关不是斜坡) 与 [第一轮复审](../../experiments/permutation_min_cost/review/REVIEW_stage1.md)。

## 5. 从单层漂移到 logits 和 benchmark

目前证据支持以下链条：

```text
跨归约块 permutation
  → down_proj 首次产生有限精度差异
  → 差异经过后续层传播
  → logits 发生小幅变化
  → 低 margin token 的 top-1 可能翻转
  → 自回归生成路径分叉
  → 最终答案或 benchmark correctness 可能改变
```

上游全模型实验中，正确的全 36 层 permutation 仍保持很高的 logits 相似度，但 top-1 一致率约为 97.8%–98.5%；翻转集中在 top-1 与 top-2 几乎并列的位置。越早的层注入漂移，传播影响越大；只动 L0 已产生接近全 36 层的大部分影响，只动最后一层则很小。

这里应区分两种推理方式：

- 当前 benchmark 主要使用**确定性 greedy 解码**。实验直接观察的是 logits 排名及 top-1/生成路径变化，并没有从 softmax 分布中随机采样。
- 如果使用随机采样，logits 改变当然也会改变 softmax 概率分布；但“采样分布如何变化、是否放大或抵消 benchmark 效应”目前没有直接实验结论。

证据见 [logits/activation 实验](../../experiments/ffn_permutation/RESULT.md) 和 [benchmark 实验](../../experiments/ffn_benchmark_eval/RESULT.md)。

## 6. 已有 benchmark 现象：Instruct 略降，Base 略升

这里比较的是 20 组随机置换：用 20 个不同随机种子各生成一套全 36 层通道布局，再分别运行同一套 benchmark。每层都对 9728 个中间通道做不重复、不遗漏的一一重排；不同层各自生成顺序，同一层的 `gate_proj`、`up_proj` 和 `down_proj` 必须正确联动，其他权重不变。这些全通道随机重排通常会跨越对齐 8 通道块；块内重排、相邻交换和整体倒序是另外的对照条件。结果如下：

| 模型 | 六任务平均分均值 ± 组间标准差 | 20 组随机置换的范围 | 当前观察 |
|---|---:|---:|---|
| Qwen3-4B-Instruct | −0.79 ± 0.19 pp | [−1.11, −0.32] pp | 20 组随机置换全为负 |
| Qwen3-4B-Base | +0.49 ± 0.21 pp | [+0.13, +0.99] pp | 20 组随机置换全为正 |

更细地看：

- MMLU、C-Eval、CMMLU 等单 token 多选任务几乎不受影响，correctness 一致率 ≥99%。
- 变化主要集中在 GSM8K、HumanEval+、MBPP+ 等长生成任务，单个 benchmark、单个 seed 可出现约 1–3 pp 的双向变化。
- Base 模型的同权重重复运行噪声地板为 0，因此其 permutation 偏移是真实的执行路径效应；Instruct 的 GSM8K 存在最高约 2.4 pp 的同权重运行噪声，需要谨慎解释，但 HumanEval+ 等无噪声任务仍证明负向效应不全是运行噪声。

机器可读汇总见 [`stage2_distribution.json`](../../experiments/ffn_benchmark_eval/results/stage2_distribution.json) 和 [`null_distribution.json`](../../experiments/ffn_benchmark_eval/results/null_distribution.json)。

## 7. 现在能说什么，不能说什么

### 已经可以说

1. Permutation 引起的有限精度漂移可以改变 logits，并在低 margin 决策处改变生成结果。
2. 这种变化不是必然退化：在当前测试中，Instruct 整体负向，Base 整体正向。
3. 方向并非跨模型统一，因此不存在“permutation 天生有利”或“天生有害”的简单规律。
4. 如果目标是严格保持原模型行为，当前最可靠方案仍是经过目标部署 kernel 验证的块内免费置换，而不是寄希望于 benchmark 平均值恰好提升。

### 目前还不能说

1. **不能把 Base 的 +0.49 pp 直接解释为模型能力提升。** 它可能是当前 benchmark 组合、greedy 路径和有限样本共同形成的稳定偏置。
2. **不能断言提升可以通过选择 permutation seed 被利用。** 事后换半数据分析中，在一半题目上获得的候选选择优势到另一半基本消失；该分析支持“额外选优不迁移”，但只覆盖当前 20 个候选和当前任务组合。
3. **不能断言漂移越大，accuracy 就单调变差或变好。** 最终轮已经补测统一的 32-prompt logits 锚点，但它与 benchmark 不一致率的相关性没有达到预注册判据。现有探针灵敏度不足，因此得到的是一次明确的判据失败，不是已建立的幅度定律。
4. **不能把 Instruct/Base 的方向差异外推到其他模型、kernel、GPU 或采样配置。**

## 8. 核心未决问题

下一阶段真正需要回答的是：

> 为什么两个模型族相对原始布局呈现相反方向，以及这种方向能否在独立模型、任务和执行栈上复现？

这个问题至少包含三个可分离部分：

1. **幅度关系**：需要怎样的探针与样本量，才能稳定预测逐题行为变化？现有 32-prompt 最后 token 锚点已经失败，不能重复当作未执行计划。
2. **方向关系**：行为变化只说明“答案变了”；变好和变坏的净差是否能跨 seed、跨样本划分稳定复现？
3. **模型差异**：为什么当前 Instruct 呈负向而 Base 呈正向？这是低-margin token 结构、训练后对齐、任务分布，还是评测/解码配置造成的？

在这三点得到验证前，最准确的总括是：**permutation 漂移确定会造成可测的行为扰动，但其 benchmark 净方向目前只是在两个具体模型和当前执行栈上的稳定现象，尚未成为可泛化、可利用的规律。**

## 9. 证据索引

| 文档 | 支持的核心内容 |
|---|---|
| [FFN permutation 原始结论](../../experiments/ffn_permutation/RESULT.md) | 三矩阵联动等价、漂移唯一来自 `down_proj`、logits/top-1/margin 结果 |
| [漂移来源分析](../../experiments/ffn_permutation/DRIFT_ORIGIN.md) | canonical-down 隔离与归约顺序证据 |
| [Benchmark 结论](../../experiments/ffn_benchmark_eval/RESULT.md) | 20 组随机置换的结果分布、Instruct/Base 方向、任务类型差异、噪声地板 |
| [Permutation 最小代价总览](../../experiments/permutation_min_cost/OVERVIEW.md) | 块宽 8、三档/两档结构、跨块不是连续旋钮、GEMM/GEMV 边界 |
| [Stage 1 复审](../../experiments/permutation_min_cost/review/REVIEW_stage1.md) | 连续几何规律失败后的阶跃结构与对齐探针 |
| [Stage 1b 复审](../../experiments/permutation_min_cost/review/REVIEW_stage1b.md) | 新 seed 预注册确认、vLLM 免费类、PyTorch GEMV 例外 |
| [v1.1 执行方失败报告](../../experiments/permutation_min_cost/FAILURE_REPORT_v1.1.md) | 1248 条正式测量、73 条失败的完整分解与复现信息 |
| [最终轮执行报告](../../experiments/noise_floor/EXECUTION_REPORT.md) | 块内端到端验收、锚点相关性失败与高斯扫描预测失败 |
| [最终轮机器验收](../../experiments/noise_floor/results/acceptance_noise_floor.json) | 硬判据、软判据和预测判据的机器可读状态 |
| [整体报告](overall_report.md) | 换半选择分析、证据等级与最终适用边界 |
