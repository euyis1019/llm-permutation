# FFN permutation 与 BF16 数值效应：整体实验报告

> 日期：2026-07-13  
> 主要模型：Qwen3-4B-Base 与 Qwen3-4B-Instruct  
> 主要环境：RTX 4090、PyTorch、vLLM、bfloat16、贪心解码  
> 阅读门槛：了解 GPU 矩阵乘法、规约，以及 permutation 是对向量或矩阵索引的重排

Base 是基础预训练模型，Instruct 是经过指令训练的版本。本文中的 BF16 与 bfloat16 指同一种 16 位浮点格式。

## 摘要

这组实验从一个简单问题开始：如果同时重排 FFN 的中间通道（intermediate channel，也常称 FFN 神经元，定义见 1.1 节）和相应权重，模型在数学上仍表示同一个函数，那么 BF16 推理是否也会给出完全相同的结果？

答案分两层。精确算术下，联动 permutation 的确不改变 FFN 函数。在真实 GPU kernel 中，`down_proj` 需要对 9728 个乘积项做规约。通道重排可能改变规约的分组和累加顺序，因此有限精度结果不一定逐比特相同。

在当前 RTX 4090、PyTorch 和 vLLM 组合上，我们找到了一类逐比特不变的置换：通道只能在各自的对齐 8 通道块内重排。实质性跨块后，输出通常进入一个由 kernel 决定的非零误差带。对 Base 模型做全 36 层随机跨块置换时，32 条 prompt 的最后一个 token logits 的 `rel_l2` 中位数是 0.009879。这里的 0.009879 是整条 logits 向量的相对 L2 距离，不能读成每个 logit 平均变化约 1%。

Benchmark 上的变化较小，主要出现在需要较长生成的任务。20 个全层随机置换中，Instruct 的六任务等权平均相对一个 baseline 为 `-0.79 ± 0.19` 个百分点，Base 为 `+0.49 ± 0.21` 个百分点。两个方向都很稳定，但现有实验只说明原始布局位于两个置换分布的不同位置，还没有解释这种位置差异来自偶然、kernel 布局偏好，还是其他机制。

高斯扰动实验发现，即使只有很小一部分 BF16 权重实际改变，logits 也能达到与全层 permutation 相近的 `rel_l2`。这是模型对微小权重变化很敏感的证据。它不是一个已经证明的通用测量下限，因为高斯扰动确实改变了参数，也改变了有限精度模型函数。现有数据还不能判断依赖前向分数的参数搜索方法会受到多大影响。

从项目进度看，FFN permutation 这一基础操作已经研究得比较完整。最初的应用目标是对齐并合并不同领域专家模型，而这一步尚未开展。当前结果能帮助下一阶段设置正确的数值对照，但不能代替真实的神经元匹配与模型合并实验。

## 1. 基础概念和指标

### 1.1 什么是 FFN 联动 permutation

FFN 是 Transformer 每层中的前馈网络模块。Qwen3-4B 的 FFN 可以简写为：

$$
h = \operatorname{SiLU}(W_gx) \odot (W_ux), \qquad y = W_dh
$$

`SiLU` 是逐元素激活函数，$\odot$ 表示逐元素乘法。中间向量 $h$ 有 9728 个维度，对应 config 中的 `intermediate_size = 9728`。本文把这个维度上的每个索引称为一个中间通道：第 $i$ 个通道由 `gate_proj` 的第 $i$ 行、`up_proj` 的第 $i$ 行和 `down_proj` 的第 $i$ 列组成，在神经元剪枝和 matching 文献中这就是一个 FFN 神经元（neuron / unit）。注意它不是 `hidden_size = 2560`，后者是 residual stream（各层之间传递的主干表示）的维度，本文的 permutation 不触碰它。取一个置换矩阵 $P$，同时修改三组权重：

$$
W'_g = PW_g, \qquad W'_u = PW_u, \qquad W'_d = W_dP^T
$$

`gate_proj` 和 `up_proj` 的行按同一顺序重排，`down_proj` 的列也跟着重排。这样一来，每个中间激活仍与原来的 `down_proj` 列配对。精确实数计算中，修改前后的 $y$ 完全相同。

只重排其中一个矩阵，或者让三个矩阵使用不同的 permutation，都会破坏这个配对关系。那属于错误操作，不是本文讨论的等价置换。

### 1.2 为什么数学相同，GPU 结果仍会不同

`down_proj` 的一个输出元素包含 9728 项乘积的求和。GPU 不会严格从第一项加到最后一项，而是先分块求部分和，再把部分和继续规约。

浮点加法不满足结合律。例如，`(a + b) + c` 与 `a + (b + c)` 可能在最后几位上不同。Permutation 没有改变参与求和的项，却可能改变这些项进入不同规约块的方式。中间舍入路径随之改变，最终 BF16 输出也可能改变。

文中的 ULP 指相邻两个可表示浮点数之间的间距。一个输出相差 1 ULP，通常只是存储值的最低有效位发生变化。它仍可能在后续 36 层传播，并在两个候选 token 分数很接近时改变贪心解码路径。

### 1.3 四个层次的等价

| 层次 | 比较的对象 | 本文如何判断 |
|---|---|---|
| 代数等价 | 精确实数函数 | 联动 permutation 前后公式相同 |
| 数值等价 | 有限精度张量 | 原始字节是否逐比特相同，或 `rel_l2` 有多大 |
| 行为等价 | 模型输出 | 最后一个 token、抽取答案、完整文本是否相同 |
| 任务等价 | Benchmark 结果 | 逐题正误（correctness）和总体准确率是否相同 |

这四层不能互相替代。完整文本不同，最终答案仍可能相同；correctness 相同，也不代表响应字节逐一相同。

还要区分三种常被统称为噪声的现象：

1. 同一权重重复推理时的运行间波动。
2. 等价 permutation 改变有限精度计算路径后产生的确定性差异。
3. 高斯扰动真实改动权重后产生的函数变化。

Base 模型在冻结配置下由两个独立进程运行，32 条 prompt 的最后一个 token logits 为 32/32 逐比特相同。这说明该探针没有观察到运行间波动。跨块 permutation 的差异则会稳定复现，因此更准确的说法是有限精度路径差异，而不是每次运行随机出现的噪声。

### 1.4 报告中的指标

`rel_l2` 定义为：

$$
\operatorname{rel\_l2} =
\frac{\lVert z_{\mathrm{case}} - z_{\mathrm{base}} \rVert_2}
     {\lVert z_{\mathrm{base}} \rVert_2}
$$

Logits 是模型在 softmax 之前为每个候选 token 给出的分数，baseline 是保留原始通道布局的参考模型。`rel_l2` 把完整 logits 张量作为一个向量比较。`rel_l2 = 0.01` 表示两条完整向量之间的 L2 距离约为 baseline 向量范数的 1%，不表示每个元素都变了 1%。

`pp` 是百分点。例如，准确率从 70% 变成 69%，变化是 `-1 pp`。六任务平均分是六项 benchmark 准确率的等权平均，不按题目数加权。表中的 `±` 是 20 组随机置换结果的总体标准差，不是置信区间。每组都由一个不同的随机种子生成通道重排布局，这些随机种子与解码采样温度无关。

`prefill` 指一次处理整段输入 prompt。`decode` 指生成阶段每次处理一个新 token。`VLLM_BATCH_INVARIANT=1` 用来尽量固定 batching 对 kernel 路径的影响。

## 2. 实验范围与证据

| 阶段 | 主要问题 | 结果位置 |
|---|---|---|
| FFN 单元测试 | 正确联动 permutation 是否满足代数等价，第一处数值差异在哪里 | [ffn_permutation/RESULT.md](../../experiments/ffn_permutation/RESULT.md) |
| Kernel 与置换结构 | 什么样的 permutation 能保持逐比特不变 | [permutation_min_cost/OVERVIEW.md](../../experiments/permutation_min_cost/OVERVIEW.md) |
| 完整模型与 benchmark | 小数值差异是否会改变 logits、生成和任务得分 | [ffn_benchmark_eval/RESULT.md](../../experiments/ffn_benchmark_eval/RESULT.md) |
| 最终预注册轮 | 验证块内置换、重复基线、锚点相关性和高斯扰动曲线 | [noise_floor/EXPERIMENT_PLAN.md](../../experiments/noise_floor/EXPERIMENT_PLAN.md) |
| 预先设计的二次分析 | 选择偏差、baseline 位置、答案投票和 NLL 尺度 | [final_round_design.md](../plans/final_round_design.md) |
| Post-hoc 补充 | 更小 sigma、FFN-only 扰动和 GSM8K 行为 | [noise_floor/SUPPLEMENT.md](../../experiments/noise_floor/SUPPLEMENT.md) |

最终一轮 `noise_floor` GPU 主实验已完成。这一轮的三个硬判据全部通过，锚点相关性判据和两条高斯扫描预测失败。

选择偏差、baseline 排名、答案投票和 NLL 尺度分析在执行前已经写入设计，后来用落盘数据计算。其中选择、baseline 和 NLL 三项没有完整实现原设计，或仍有统计口径问题。更小 sigma、FFN-only 扰动和 GSM8K 行为这三个补充臂是在看到主结果后追加的，证据权重低于预注册结果。

## 3. 实验结果

### 3.1 对齐 8 通道块内的置换可以逐比特不变

`对齐 8 通道块` 指 `[0, 7]`、`[8, 15]`、`[16, 23]` 这样的固定分组。通道可以在自己的 8 元组内任意换位，但不能移到相邻 8 元组。

更早的 `permutation_min_cost` v1.1 实验使用了另一套硬判据。它最初把 PyTorch GEMV 也纳入逐比特适用域，因此该轮硬判据失败并按计划停机。复审把 GEMV 单列为例外后，保留了三个适用组合：

| 计算路径 | 结果 |
|---|---:|
| PyTorch BF16 GEMM，输入行数至少为 2 | 126/126 逐比特相同 |
| vLLM batch-invariant prefill | 126/126 逐比特相同 |
| vLLM batch-invariant decode | 126/126 逐比特相同 |
| 合计 | 378/378 逐比特相同 |

这一结果与下面的解释一致：当前 kernel 的规约结构会保留这些 8 元组。块内改变顺序时，每个规约块仍拿到同一组乘积项，因此部分和没有变化。

这个规律有明确边界。PyTorch 在输入只有一行时会走 GEMV 路径，126 个块内置换中只有 53 个逐比特相同。其余 73 个案例每次只有 2560 个输出元素中的 1 到 3 个发生变化，`rel_l2` 位于 `1.1e-9` 到 `6.0e-5`。这说明 GEMV 使用了不同的规约结构。

全模型验证也基本一致。Base 模型的 36 层全部做块内置换后，32 条 prompt 的最后一个 token logits 为 32/32 逐比特相同，六项 benchmark 的逐题 correctness 也全部相同。GSM8K 的 500 条样本中，有 497 条响应文本逐字节一致；另外 3 条文本不同，但 correctness 没变。其余五项的响应文本全部一致。当前证据支持 logits 探针和 correctness 层的等价，不支持所有生成文本都能严格逐字节复现。

块宽 8 是当前 RTX 4090、dtype、kernel 版本和输入形状的实测属性，不是模型数学结构中的自然常数。更换 GPU、PyTorch、vLLM、kernel 或 dtype 后，应先跑一个小规模逐比特 smoke test。

### 3.2 实质性跨块置换会进入 kernel 相关的误差带

在 PyTorch BF16 GEMM 中，实质性跨越对齐 8 块后，单层 `down_proj` 的 `rel_l2` 通常落在 `2.4e-3` 到 `3.2e-3`。把 5% 的通道散置到其他块，与把 50% 的通道散置到其他块，常常得到相近结果。对这个 kernel 来说，单层跨块更接近开关，而不是一条随移动比例平滑上升的曲线。

这里仍有一些过渡案例，结果依赖数据和 kernel。vLLM 的表现也不同，其非零结果大致分布在 `2e-5` 到 `2e-4`，没有 PyTorch GEMM 那样紧的 `3e-3` 平台。因此，`跨块后立刻到 2.9e-3` 只能描述已测的 PyTorch GEMM 主体区域，不能外推到所有实现。

在完整 Base 模型上，三个代表条件的 32-prompt 最后 token logits 结果是：

| 置换条件 | 中位 `rel_l2` | 逐比特相同的 prompt 数 |
|---|---:|---:|
| 36 层对齐 8 块内重排 | 0 | 32/32 |
| 36 层奇数起点的相邻交换 | 0.009489 | 0/32 |
| 36 层全局随机置换 | 0.009879 | 0/32 |

奇数起点交换使用 `(1,2)`、`(3,4)` 这样的配对，其中 `(7,8)`、`(15,16)` 等配对会跨越对齐 8 块的边界。

后文把 `0.009879` 称为 permutation 参考线。它是当前模型、栈、prompt 集和指标下的操作性参照，不是所有模型共有的常数，也不是同权重重复运行的随机噪声。

### 3.3 Benchmark 变化较小，主要集中在生成任务

这里的 20 组随机置换，指用 20 个不同随机种子各生成一套模型通道布局。随机只决定通道的新顺序，置换本身始终遵守以下规则：

1. 36 层 FFN 全部参与置换，每层各自生成一套不同的随机顺序。
2. 每层都对 9728 个中间通道做完整的一一重排。每个通道恰好出现一次，不会遗漏或重复。
3. 同一层内，`gate_proj` 和 `up_proj` 的行与 `down_proj` 的列使用完全相同的顺序。注意力层、embedding、归一化层和其他权重保持不变。

第三条就是前文所说的联动置换。它保证每个中间通道仍与原来的输出权重配对。随机置换不是随意改动三块矩阵，也不是三块矩阵各自独立打乱。

这 20 组随机置换不要求通道留在原来的对齐 8 通道块内，因此通常会跨越这些块。块内重排、相邻交换和整体倒序是另外设置的对照条件，不属于这里的 20 组随机置换。

每组置换保存为一个权重快照（checkpoint），再评测 MMLU、GSM8K、C-Eval、CMMLU、HumanEval+ 和 MBPP+。其中 MMLU、C-Eval 和 CMMLU 主要做短答案的多选决策，GSM8K 和两个代码任务需要较长生成。

| 模型 | 六任务平均分相对 baseline 的变化 | 20 组随机置换的方向 |
|---|---:|---:|
| Qwen3-4B-Instruct | `-0.79 ± 0.19 pp` | 20/20 为负 |
| Qwen3-4B-Base | `+0.49 ± 0.21 pp` | 20/20 为正 |

MMLU、C-Eval 和 CMMLU 的逐题 correctness 一致率超过 99%。这很符合 margin 直觉：如果第一名和第二名 token 的分数差距较大，末位数值变化不会改变最大项。

长生成更容易出现分叉。某一步遇到两个分数接近的 token 时，微小 logits 差异可能改变贪心选择。新 token 又会成为下一步输入，之后的文本便可能走向另一条路径。实验中的分数变化也主要集中在 GSM8K、HumanEval+ 和 MBPP+。

这些变化需要结合重复基线看。Base 的 11 次同权重 benchmark 重跑在 correctness 上完全一致。Instruct 的残余波动主要来自 GSM8K，其同权重重跑的单项差值最高达到 2.4 pp，六任务平均的范围为 0.4 pp。因此，Base 上的 permutation correctness 差异不是普通运行间波动；Instruct 的 GSM8K 则存在混淆，不能把全部差值都归给 permutation。

### 3.4 两个模型各自同号，但彼此方向相反

20 个置换布局形成的分布与原始 baseline 的位置如下：

| 模型 | Baseline 六任务平均分 | 20 组随机置换的均值与标准差 | 观察范围 | Baseline 位置 |
|---|---:|---:|---:|---|
| Base | 72.30 | `72.79 ± 0.21` | 72.43 到 73.29 | 低于全部 20 组随机置换 |
| Instruct | 72.33 | `71.54 ± 0.19` | 71.22 到 72.01 | 高于全部 20 组随机置换 |

这张表能解释为什么相对 baseline 计算时，Base 的 20 个差值都为正，而 Instruct 都为负。它不能解释 baseline 为什么会落在分布的一端。

把这个现象直接称为回归均值会说得太满。Baseline 排名分析只做了准确率分布、排名和同权重重跑范围，没有完成原设计中的逐题幸运与不幸分解，也没有比较各 seed 的翻转重叠。原始未置换布局还可能与当前 kernel 有系统性的相互作用。现有数据不支持置换必然伤害 Instruct 这一普遍结论，也没有排除更具体的布局机制。

### 3.5 平均选择优势在验证半基本消失

选择分析每次随机把各 benchmark 的题目分成两半。在 A 半上，从 20 个置换模型中挑选六任务平均分最高的一个，再查看同一个模型在 B 半上的分数。这个随机划分重复了 200 次。

| 模型 | A 半相对 20 组随机置换的均值 | B 半相对 20 组随机置换的均值 | B 半相对 baseline |
|---|---:|---:|---:|
| Base | `+0.631 pp` | `+0.070 pp` | `+0.562 pp` |
| Instruct | `+0.617 pp` | `-0.058 pp` | `-0.853 pp` |

A 半上约 0.6 pp 的选择优势，在 B 半上相对候选池均值接近于零。候选越多，最高分越容易包含一次有利的样本波动，换一批题后这部分通常不会保留。

B 半相对 baseline 的差值并没有消失，因为上一节中的整个候选池偏移仍然存在。Base 候选池整体高于 baseline，Instruct 候选池整体低于 baseline。这个实验支持选出的额外优势不迁移，不能证明候选池与 baseline 之间的差异都来自抽样误差。

实际分析做了 200 次随机半分和 best-1 选择，没有按原设计做固定奇偶划分及 `k = 1, 2, 3, 5` 曲线。这 200 次划分共享同一批题和同一组模型，不是 200 个独立实验。`约 0.6 pp` 只适用于当前 20 候选、半数据选优和六任务平均的设置。

### 3.6 高斯扰动与 permutation 处在相近的 logits 尺度

主实验向 Base 模型全部浮点参数加入高斯噪声。噪声直接以原 BF16 dtype 原位相加，因此一部分很小的加法会被舍掉。`权重实际改变比例` 指 BF16 存储值确实发生变化的元素比例，而不是理论上采样过噪声的比例。

| sigma | BF16 权重实际改变比例 | 权重 `rel_l2` | Logits 中位 `rel_l2` | 相对 permutation 参考线 |
|---:|---:|---:|---:|---:|
| `1e-6` | 1.31% | `8.02e-6` | 0.00994 | 1.01 倍 |
| `3e-6` | 3.81% | `4.02e-5` | 0.01038 | 1.05 倍 |
| `1e-5` | 11.69% | `2.29e-4` | 0.00992 | 1.00 倍 |
| `3e-5` | 30.23% | `1.05e-3` | 0.01194 | 1.21 倍 |
| `1e-4` | 64.35% | `4.06e-3` | 0.01561 | 1.58 倍 |
| `1e-3` | 95.69% | `3.89e-2` | 0.10071 | 10.19 倍 |
| `1e-2` | 98.96% | `3.89e-1` | 1.08241 | 109.57 倍 |

曲线上最清楚的现象出现在 `sigma = 1e-6` 到 `1e-5`：实际改动的 BF16 权重比例明显上升，logits `rel_l2` 仍接近全层随机 permutation 的参考线。到 `1e-4` 后，曲线才开始持续离开这个尺度。

不过，这不等于前向计算存在一个已证明的 1% 随机噪声地板。Permutation 在精确算术下保持函数不变，高斯扰动则真实改动了数千万到数十亿个 BF16 参数。两者得到相近的 logits 距离，可能反映深层网络对微小局部变化的放大，也可能包含量化和 kernel 路径的共同作用。当前实验没有把这些来源分开。

预注册的平台预测要求同时看到一段与 permutation 同量级的平台，以及平台下方更小 sigma 的逐比特零区。原判据没有识别出正式的平台段，实验在最低主网格 `1e-6` 也已经得到非零结果。权重量化吞噬本身仍然存在，因为多数参数在小 sigma 下没有改变；失败的是更小档会全部被吞掉这一完整预测。

### 3.7 当前实验还没有测出零阶优化的信噪比

RandOpt 会生成多个参数扰动候选，再用前向任务分数选择候选。MeZO 和 FZOO 一类零阶方法不做反向传播，而是从参数扰动前后的损失函数差分估计方向。典型的双边探针形式是：

$$
\frac{L(\theta + \epsilon u) - L(\theta - \epsilon u)}{2\epsilon}
$$

其中 $\theta$ 是模型参数，$u$ 是扰动方向，$\epsilon$ 是步长，$L$ 是损失。要判断这个估计是否被有限精度效应干扰，至少需要让 `+u` 与 `-u` 使用同一数据、同一参数范围和配对执行，并统一 loss 的聚合方式。

现有 NLL 尺度分析没有做这项实验。NLL 是负对数似然，越低表示模型给正确后续文本的概率越高。分析把 permutation 的逐 prompt `|delta NLL|` 均值，与高斯扰动的 32-prompt 有符号 `delta NLL` 先平均再取绝对值进行比较，分子和分母不是同一个统计量。

高斯权重快照使用独立的单边随机方向，也不是同一方向的 `theta + epsilon u` 与 `theta - epsilon u`。所以，结果文件中 `1e-4` 为 1.58 倍、`1e-3` 为 6.54 倍的比值不能用作 MeZO 或 FZOO 的信噪比结论。

下一步要直接测量：当真实探针信号与 permutation 参考线相近时，候选排序是否稳定？小于参考线的真实效应仍可能通过重复、配对和更多样本估计出来，不能一律叫作噪声。

### 3.8 GSM8K 的答案投票高于单模型

答案投票分析对 20 个置换模型的抽取答案做 plurality vote，也就是选择票数最多的答案。

| 模型 | 投票 | Baseline | 最佳单 seed |
|---|---:|---:|---:|
| Base | 79.4% | 77.2% | 79.2% |
| Instruct | 76.8% | 75.6% | 74.6% |

目前只有 GSM8K 这一个 benchmark 出现投票高于 baseline 和最佳单 seed 的结果，但 Base 与 Instruct 两个模型族都观察到了。Instruct 的 GSM8K 同权重重跑波动最高可达 2.4 pp，六任务平均的投票分数也仍低于其 baseline，所以这个结果不能扩展为整个 benchmark 套件的集成收益。

这项分析没有独立验证集或显著性检验，平票还会受 seed 顺序影响。运行 20 个模型需要大约 20 倍前向计算，省掉的是额外训练，不是推理成本。

代码任务不能直接照搬字符串投票。两段功能相同的代码可以使用不同变量名和结构，整段响应字符串不相同并不代表不能形成有效集成。

## 4. 看到主结果后追加的三个实验

这三个实验在主结果已经可见后才设计，因此属于 post-hoc 探索。它们扩展了观察范围，不替代预注册验证。

| 补充臂 | 核心观测 | 判断 |
|---|---|---|
| S1，更小 sigma | `sigma = 1e-8` 时约 53 万个参数改变，3 个 seed 的 logits `rel_l2` 为参考线的 0.89 到 1.00 倍 | 已测范围内仍没找到逐比特零区 |
| S2，只扰动 FFN | `1e-6` 和 `1e-5` 的结果为参考线的 0.87 到 1.07 倍 | 相近尺度并非只由非 FFN 参数造成 |
| S3，GSM8K-500 | `1e-4` 的 5 个 seed 都在 76.2% 到 79.2% 内，`1e-3` 有 2/5 低于这个范围 | 只是范围比较，不是统计等价检验 |

S1 中，53 万个变化占 40.22 亿参数的 0.013%，约等于每 7500 个权重中有一个变化。3 个 seed 的最后 token top-1 各有 0 到 1 条翻转。这说明极稀疏的实际权重变化经过模型后可以对应较大的 logits 距离，不说明只改一个参数也会触发完整参考带。

S3 中 `落在范围内` 只表示样本没有超过 20 个置换 seed 的最小值和最大值。这个范围不是置信区间。`sigma = 1e-4` 的五次均值约为 77.52%，而 permutation 分布均值是 78.17%，两者是否等价仍需正式检验。

## 5. 三条使用规则

1. 对齐 8 块是部署属性。它在三个保留的 kernel 与形状组合中得到 378/378 验证，PyTorch T=1 GEMV 已经构成反例。换 GPU、kernel 或 dtype 时要重测。
2. `0.009879` 是本实验的 permutation 参考线。它不等于所有参数扰动方法共有的精度下限。高斯实验真实改动了权重，零阶方法的信噪比也尚未直接测量。
3. Benchmark 结论需要独立验证。约 0.6 pp 的选择偏差只对应当前 best-of-20 设置，Base 与 Instruct 的相反方向仍缺少机制解释，GSM8K 投票也只是一个待复现的单任务结果。

## 6. 下一步实验

### 6.1 部署中的通道重排

可以先探测当前部署 kernel 中保持逐比特不变的块宽，再尽量把 permutation 限制在块内。当前栈的块宽是 8。这个限制适合作为零差异正控制，但它只能在每个 8 元组内匹配通道，无法完成任意两个 expert 之间的全局神经元对齐。

### 6.2 参数扰动的比较和选择

候选评估需要独立的选择集和验证集。还应同时记录重复 baseline、logits margin、逐 token 首次分叉和最终 correctness。名义上的 sigma 不够，还要报告实际改变的 BF16 参数比例与权重 `rel_l2`。

零阶方法需要直接执行配对的 `theta + epsilon u` 与 `theta - epsilon u` 探针。两边必须共享方向、prompt、参数范围和执行配置，loss 应使用统一的 FP32 或 FP64 累积口径。只有这样才能估计方向信号与有限精度差异的比例。

### 6.3 Expert 通道匹配与权重合并

这里的 expert 指从同一个基础模型出发，经过不同领域训练或剪枝得到的模型。Matching 是先找出两个 expert 中功能相近的中间通道，再把它们放进同一坐标顺序；merge 则在对齐后合并权重。Naive merge 指不做这一步，直接按当前位置平均。

下一项直接回答原应用目标的实验，是选择两个同架构、来自同一 base 的 expert，完成 matching 和 merge。匹配特征可以使用 `gate_proj` 行、`up_proj` 行、`down_proj` 列以及校准集激活。对照组至少要包含 naive merge、随机匹配、权重匹配和激活辅助匹配。

这组 permutation 实验提供了代数正确性检查、有限精度对照和 kernel 边界。它尚未证明对齐后合并会优于直接合并，也没有覆盖 MoE。MoE 会在多组 FFN expert 之间用 router 选择计算分支，重排 expert 时还要同步处理 router 的索引。

## 7. 证据文件

主实验脉络集中在三份文档：[FFN permutation 单元与全模型结果](../../experiments/ffn_permutation/RESULT.md)、[permutation 最小代价总览](../../experiments/permutation_min_cost/OVERVIEW.md) 和 [benchmark 结果及相邻交换勘误](../../experiments/ffn_benchmark_eval/RESULT.md)。

最终轮的冻结设计、执行摘要和机器验收分别见 [EXPERIMENT_PLAN.md](../../experiments/noise_floor/EXPERIMENT_PLAN.md)、[EXECUTION_REPORT.md](../../experiments/noise_floor/EXECUTION_REPORT.md) 与 [acceptance_noise_floor.json](../../experiments/noise_floor/results/acceptance_noise_floor.json)。验收文件保留了锚点相关性失败和两条高斯扫描预测失败的记录。

Reviewer 分析产物包括 [选择分析](../../experiments/noise_floor/reviewer_analysis/part3.json)、[baseline 排名](../../experiments/noise_floor/reviewer_analysis/part4.json)、[答案投票](../../experiments/noise_floor/reviewer_analysis/part5.json) 和 [NLL 尺度比较](../../experiments/noise_floor/reviewer_analysis/part7_zo_floor.json)。Post-hoc 的设计与结果在 [SUPPLEMENT.md](../../experiments/noise_floor/SUPPLEMENT.md) 和 [supp_summary.json](../../experiments/noise_floor/reviewer_analysis/supp_summary.json)。
