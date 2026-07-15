# 2025–2026 随机参数扰动训练：从 Zeroth-Order 到 RandOpt / Evolution Strategies

> 检索截止：2026-07-12。范围以 LLM 为主，补充大视觉/多模态模型。本文优先解读有公开实现的工作，并严格区分“扰动权重”“扰动重参数化变量”和“只扰动输入/latent/token”的方法。

> 配套代码抽象：[`minimal_perturbation_framework/`](./minimal_perturbation_framework/README.md)。它把本文方法统一成 `proposal → black-box score → deployment plan → commit`，并保留权重更新、committee 与蒸馏三种不同输出。

## 先读这个：方法索引

第一次读这条文献线时，先不要被缩写带着走。所有核心方法的前半段几乎一样：**在当前模型附近制造参数候选，然后只用 forward 得到 loss 或 reward**。真正区分它们的是拿到分数之后做什么：

| 家族 | 拿到候选分数之后做什么 | 先记住的代表方法 |
|---|---|---|
| **ZO（zeroth-order，零阶优化）** | 用少量正反试探的分数差，猜一个近似更新方向，然后移动当前模型。 | MeZO、DiZO、FZOO、SubZero、LOZO |
| **ES（evolution strategies，进化策略）** | 一次评估一群候选，用整个人群的 reward 加权产生下一轮中心。 | ES-at-Scale、EGGROLL、QES |
| **邻域专家搜索/整合** | 不一定估梯度；先筛出预训练模型附近的专家，随后保留 ensemble 或把群体合回单模型。 | RandOpt、CoRP |
| **系统实现** | 不改变“怎么估方向”，而是重排模型、数据与 forward 的执行方式。 | ZO2、DistZO2 |

> **最容易混淆的一组：** `ZO2 / DistZO2` 优化的是运行系统；`QuZO` 用低比特参数和低比特扰动做两点 ZO；`QZO` 主要冻结整数权重、改调连续 quantization scale；`QES` 则直接在离散整数权重空间运行 population ES。

下面按推荐阅读顺序列出核心方法。每项只回答三个问题：它叫什么、直觉上做什么、创新究竟在哪里。

**推荐阅读目录（点击方法名进入 arXiv）：**

1. [MeZO](https://arxiv.org/abs/2305.17333) — 先建立两点 ZO 和 seed replay 的基本直觉。
2. [DiZO](https://arxiv.org/abs/2502.03304) — 看 MeZO 的更新如何做逐层整形。
3. [FZOO](https://arxiv.org/abs/2506.09034) — 看如何把多个扰动并成 GPU batch、减少 forward 次数。
4. [SubZero](https://arxiv.org/abs/2410.08989) — 看如何把搜索限制在周期性刷新的随机低秩子空间。
5. [LOZO](https://arxiv.org/abs/2410.07698) — 看如何直接设计低秩扰动和低内存 momentum。
6. [SensZOQ](https://arxiv.org/abs/2406.02913) — 看极少量敏感参数与量化如何结合。
7. [HiZOO](https://arxiv.org/abs/2402.15173) — 看曲率信息如何帮助不同方向采用不同力度。
8. [ZO Fine-tuner](https://arxiv.org/abs/2510.00419) — 看扰动策略如何从手工规则变成可学习优化器。
9. [ZO2](https://arxiv.org/abs/2503.12668) / [DistZO2](https://arxiv.org/abs/2507.03211) — 看不改估计器时，系统调度如何降低 GPU 显存门槛并扩展到多卡。
10. [QuZO](https://arxiv.org/abs/2502.12346) — 看低比特参数和低比特扰动中的两点 ZO。
11. [QZO](https://arxiv.org/abs/2505.13430) — 看如何冻结整数权重码、转而优化连续量化尺度。
12. [ES-at-Scale](https://arxiv.org/abs/2509.24372) — 从单个方向试探过渡到全参数 population ES。
13. [EGGROLL](https://arxiv.org/abs/2511.16652) — 看低秩扰动如何服务于大 population 的硬件吞吐。
14. [QES](https://arxiv.org/abs/2602.03120) — 看 population ES 如何直接运行在离散整数权重空间。
15. [Neural Thickets / RandOpt](https://arxiv.org/abs/2603.12228) — 看随机扰动如何从“估方向”变成“筛邻域专家”。
16. [CoRP](https://arxiv.org/abs/2605.31494) — 看如何把 rewarded experts 合并回单个可部署模型。

### 1. MeZO — Memory-Efficient Zeroth-Order Optimizer（ZO 基线）

- **讲解：** [MeZO](https://arxiv.org/abs/2305.17333) 沿同一个随机方向分别试探正反两边，比较两次 loss，再判断应该往哪边更新。它只保存随机种子，需要噪声时逐层重新生成，因此不必保存反传 activation、梯度或一份完整扰动。

- **创新点：** 它没有发明两点 ZO，而是把经典方法改造成可原位执行、可 seed replay 的 LLM 全参数微调流程，使训练显存接近推理。

### 2. DiZO — Divergence-driven Zeroth-Order Optimization（逐层整形）

- **讲解：** [DiZO](https://arxiv.org/abs/2502.03304) 认为普通 MeZO 容易让各层以不合适的相对幅度移动。它保留正反两次试探，但以预训练权重为锚点，周期性判断不同层的累计更新应该放大还是缩小。

- **创新点：** 它的外层估计器仍是 MeZO；真正的新意是 anchor-based、divergence-driven 的逐层投影与尺度控制。

### 3. FZOO — Fast Zeroth-Order Optimizer（把 ZO 做成 GPU batch）

- **讲解：** [FZOO](https://arxiv.org/abs/2506.09034) 不再依次跑一个方向的正负两边，而是把原模型和一批单边扰动候选放在一起评估，再统一与原模型 loss 比较。它根据这批 loss 的离散程度自动归一化更新，并使用只有正负一的 Rademacher 噪声方便 GPU 并行。

- **创新点：** 把 batched one-sided estimator、自适应步长和 GPU 友好的扰动结合起来，重点减少收敛所需的 forward 次数和逐方向执行开销。

### 4. SubZero — random Subspace Zeroth-order optimization（随机子空间）

- **讲解：** [SubZero](https://arxiv.org/abs/2410.08989) 不在整个参数空间里乱试，而是为权重建立一个小型随机低秩子空间，只在其中做正反试探。这个子空间隔一段时间刷新，也能套在 full tuning、LoRA、prefix 和 prompt tuning 外面。

- **创新点：** 用“若干步内复用、周期性刷新”的随机低秩子空间降低有效搜索维数和估计方差。

### 5. LOZO — low-rank ZO algorithm（直接做低秩扰动）

- **讲解：** [LOZO](https://arxiv.org/abs/2410.07698) 假设 LLM 微调的有用矩阵更新往往是低秩的，因此直接把每次矩阵扰动写成两个瘦矩阵的乘积。它还能以压缩形式保存 momentum，而不需要模型大小的动量状态。

- **创新点：** 设计直接匹配低秩梯度结构的 ZO estimator，并给出几乎不增加内存的 momentum 版本。它与 SubZero 的区别是：SubZero 强调随机子空间，LOZO 强调矩阵级低秩 estimator。

### 6. SensZOQ — Sensitive ZO optimization with Quantization（极稀疏目标微调）

- **讲解：** [SensZOQ](https://arxiv.org/abs/2406.02913) 先从预训练或源任务梯度中找出一张可迁移的敏感参数 mask；到目标设备后，只用 ZO 更新约 0.1% 的高精度敏感参数，其余权重量化并冻结。

- **创新点：** 把可跨任务复用的静态敏感 mask 与量化结合，显著缩小目标阶段的 ZO 搜索空间。要注意，mask 的构造依赖源阶段的一阶梯度，只有目标微调阶段是 ZO。

### 7. HiZOO — Hessian-Informed Zeroth-Order Optimizer（曲率感知）

- **讲解：** [HiZOO](https://arxiv.org/abs/2402.15173) 认为不同参数方向有的陡、有的平，不应该都用相同力度。它多做一次 forward 来估计对角曲率，再据此重新缩放 ZO 更新。

- **创新点：** 在 LLM 的 forward-only 微调中引入对角 Hessian 预条件，以一个额外 forward 换取更适合不同曲率方向的步幅。

### 8. ZO Fine-tuner（让模型学习“该怎么扰动”）

- **讲解：** [ZO Fine-tuner](https://arxiv.org/abs/2510.00419) 不再为所有参数手工固定同一种扰动尺度，而是训练一个很小的学习器，为不同参数组预测不同的扰动强度。这个学习器针对一个 base LLM 训练一次，之后可以在该基座的多个任务和衍生 checkpoint 上复用。

- **创新点：** 把 ZO 的扰动策略从人工规则变成可 meta-learn、非均匀且自适应的 optimizer，并尝试把前置学习成本摊销到多个下游任务。

### 9. ZO2 / DistZO2 — Zeroth-Order Offloading（系统工作）

- **讲解：** [ZO2](https://arxiv.org/abs/2503.12668) 仍然运行普通 MeZO 式的两次 forward，但把 Transformer block 主要放在 CPU，需要时逐块搬进 GPU，并重叠搬运与计算；它省的是峰值 GPU 显存，代价会转移到 CPU RAM 和互联。[DistZO2](https://arxiv.org/abs/2507.03211) 再把正负扰动和数据 batch 分给多张 GPU 并行。

- **创新点：** ZO2 围绕“双 forward”重做 CPU/GPU offload 调度；DistZO2 则加入 perturbation parallelism、面向 ZO 的 DDP 与二维并行。它们没有提出新的梯度估计器。

### 10. QuZO — Quantized Zeroth-Order Fine-Tuning（低比特两点 ZO）

- **讲解：** [QuZO](https://arxiv.org/abs/2502.12346) 让量化模型直接在 4/8-bit forward 环境中做两点 ZO，随机扰动本身也被压成低比特。全参数版本会更新低比特模型参数，并不是只调 quantization scale。

- **创新点：** 为同一随机方向设计独立 stochastic rounding，使量化扰动带来的额外偏差在平均意义上被抵消，从而不依赖反传里的 straight-through estimator。

### 11. QZO — Quantized Zeroth-order Optimization（改调量化尺子的刻度）

- **讲解：** [QZO](https://arxiv.org/abs/2505.13430) 主要冻结离散整数权重码，改为扰动把整数码还原成有效权重时使用的连续 quantization scale。可以把它理解成“格点不动，只调整尺子的单位长度”，因此细小更新不会直接被整数舍入吞掉。

- **创新点：** 把 ZO 重参数化到连续 scale 空间，并用 directional-derivative clipping 压住异常方向信号、降低方差。

### 12. ES-at-Scale — Evolution Strategies at Scale（全参数群体更新）

- **讲解：** [ES-at-Scale](https://arxiv.org/abs/2509.24372) 每轮在当前 LLM 周围采样一群全参数候选，让它们分别完成任务并获得 reward；高分扰动对下一轮中心的移动贡献更大。它不对 rollout 反传，所以可以直接使用代码测试、规则判分器或 exact-match 等不可微 reward。

- **创新点：** 它没有发明 ES，而是证明不借 LoRA 或降维，直接在十亿级 LLM 的完整参数空间运行 ES 也能奏效，并可把主要 workload 交给并行推理系统。

### 13. EGGROLL — Evolution Guided GeneRal Optimisation via Low-rank Learning（硬件友好的 ES）

- **讲解：** [EGGROLL](https://arxiv.org/abs/2511.16652) 把每个候选的巨大随机矩阵拆成两个瘦矩阵，使单个扰动保持低秩并能用高效矩阵乘法执行。每个候选虽然是低秩的，但整个人群的加权组合仍可形成复杂、高秩的最终更新。

- **创新点：** 它把低秩首先当成 GPU 算术强度和大 population 吞吐问题来解决，使 ES 更接近 batched inference，而不是把模型永久限制在一个固定 LoRA 空间。

### 14. QES — Quantized Evolution Strategies（离散整数空间的 ES）

- **讲解：** [QES](https://arxiv.org/abs/2602.03120) 直接在量化整数权重格点上生成一群候选、用 reward 排分，再汇成下一次整数更新。若一次更新小得不足以跨过一个格点，它会先保留“小数余量”，以后攒够再让整数权重跳一格。

- **创新点：** 把 accumulated error feedback 与 stateless seed replay 结合，让离散格点不再吞掉长期微小信号，同时避免常驻一份高精度模型状态。

### 15. RandOpt — Random Optimization（从邻域里筛专家）

- **讲解：** [Neural Thickets](https://arxiv.org/abs/2603.12228) 是论文和“预训练权重附近存在稠密、多样任务专家”这一现象的名字；**RandOpt 才是算法名**。它固定预训练中心，独立采样大量权重扰动，用 support set 选出 top-k，测试时让入选专家分别回答并投票。

- **创新点：** 它把随机扰动从“估计更新方向的探针”改成“寻找附近已有专家的候选”。原始 RandOpt 不估梯度、不移动中心，最终产物是一组专家，而不是一个收敛后的单模型。

### 16. CoRP — Consolidating Rewarded Perturbations（把专家群压回单模型）

- **讲解：** [CoRP](https://arxiv.org/abs/2605.31494) 针对 RandOpt 推理时需要运行 top-k 次的问题，把高 reward 且彼此兼容的扰动在权重空间中合成一次更新，削弱互相冲突的方向。合并出的模型还要通过 held-out validation gate，确实改善才接纳并作为下一轮采样中心。

- **创新点：** 把 RandOpt 的 prediction-level ensemble 变成 weight-level population collapse，利用 rewarded population 中可复现的低秩结构，恢复单模型、单次 forward 的部署方式。

### 后文其他名字速查：第一遍可以先跳过

| 名字 | 几句话理解 | 创新点或本文定位 |
|---|---|---|
| [The Blessing of Dimensionality](https://arxiv.org/abs/2602.00170) | 解释为什么十亿维模型里只采几十个 ES 方向仍可能有用：真正影响 reward 的高曲率方向可能很少。 | **解释论文，不是新优化器。** |
| [AGZO](https://arxiv.org/abs/2601.17261) | 从 forward activation 中提取低秩基，只在 activation 指示的权重更新子空间里扰动。 | 用 activation 动态引导权重扰动；不是 activation noise。 |
| [AdaLeZO](https://arxiv.org/abs/2604.18264) | 把“这一步该扰动哪一层”当成随训练变化的 bandit 问题。 | 自适应分配逐层查询预算，并用概率校正保持估计可靠。 |
| [ZO-Act](https://arxiv.org/abs/2607.01125) | 初始化时只分析一次 activation，建立固定低秩 basis；之后冻结低比特基座，只用 ZO 调小系数矩阵。 | 与 AGZO 的区别是 one-shot basis 和显式低维重参数化。 |
| [P-GAP](https://arxiv.org/abs/2510.18228) | 先用 ZO probes 估一个低维近似梯度空间，再让后续扰动与这个方向对齐。 | 用 gradient-aligned perturbations 减少无效随机试探。 |
| [RoZO](https://aclanthology.org/2026.eacl-long.80/) | 把固定秩 LoRA adapter 看成带几何约束的空间，只在合法切空间里探测和更新。 | 将 Riemannian geometry、状态搬运和 trust region 引入 adapter ZO。 |
| [MoZO](https://arxiv.org/abs/2506.12409) | 面向视觉语言持续学习，在不同 modality branch / layer 之间混合 FO 与 ZO，并稳定高方差视觉扰动。 | 贴题但目前无公开代码，正文只作 roadmap。 |
| [RLR Optimizer](https://arxiv.org/abs/2502.00639) | Recursive Likelihood Ratio Optimizer；递归组合 likelihood-ratio、ZO 和 FO estimator 来对齐 diffusion。 | **混合 half-order 方法，不是纯参数扰动替代反传。** |
| [ZOO-Prune](https://arxiv.org/abs/2509.24837) | 给轻量 projection 加噪，观察输出变化来判断哪些视觉 token 重要。 | **用于 training-free token pruning，不更新模型权重。** |
| [Dual-Seed Evolutionary Algorithm](https://ojs.aaai.org/index.php/AAAI/article/view/37893) | 进化搜索 diffusion 的输入 noise seed。 | 扰动的是输入 seed，不是 parameter matrix。 |
| [GS-ES](https://openreview.net/forum?id=McVjYBWMpT) | 对 denoising action 做 ES-inspired sampling，最后仍回传 fitness-weighted gradients。 | hybrid roadmap，不是纯 gradient-free 权重 ES。 |
| [Iterative RandOpt](https://github.com/sunrainyg/RandOpt) | 多轮小 population 搜索后，用 SFT/KD 把专家知识蒸馏回单模型，再重新居中。 | 目前是 RandOpt 仓库分支而非独立论文；完整循环使用反传。 |

## 0. Executive summary

这一波方法的共同动作很朴素：不对计算图做反向传播，而是在参数点 `θ` 附近采样扰动 `ε`，运行模型拿到 loss/reward，再决定保留哪些候选或往哪个方向移动。但它们解决的是三类不同问题：

1. **两点 ZO / MeZO 家族**：比较 `θ+σε` 和 `θ-σε`，用损失差把随机方向变成一个近似更新方向；目标通常是以接近推理的显存做监督微调。
2. **ES 家族**：一次评估一群参数候选，用群体 fitness 加权合成下一步更新；更适合不可微 reward、离散权重和高度并行的 post-training。
3. **RandOpt 家族**：不急着迭代更新；从预训练点附近大规模采样，直接留下各有所长的专家。原始 RandOpt 的关键产物是 top-k ensemble，不是一个收敛后的单模型。

2025–2026 最显著的变化，是研究问题从“ZO 能否省显存”转向：如何控制随机估计方差（低秩/敏感子空间/按层缩放）、如何把推理硬件吃满（batch、offload、seed replay），以及为何十亿参数空间里很小的 population 仍然有效。若只挑最值得读代码的主线，建议顺序是：

`MeZO 背景 → DiZO / FZOO / SubZero → ES at Scale → EGGROLL → QES → RandOpt → CoRP`。

## 1. Background：为什么用扰动代替反传

反向传播不仅要运行模型，还要保留 activation、gradient 和 optimizer state。模型越大，这些训练态内存越容易成为瓶颈。随机参数扰动只需要若干次 forward：给定目标函数 `f`，最经典的两点估计在直觉上就是：

```text
抽一个随机方向 ε
分别测 f(θ + σε) 与 f(θ - σε)
若 +ε 更好，就沿 +ε 走；若 -ε 更好，就反向走
```

随机种子可以重建 `ε`，因此无需常驻一份与模型等大的扰动张量。不过，“不存反传状态”不等于“计算便宜”：每一步至少多次 forward，随机方向的方差也可能让总步数上升。最新工作大多是在处理这个交换关系。

还有一个容易忽略的变化：监督学习里的 `f` 通常是可微 loss；LLM post-training 里的 `f` 可以是 exact-match、代码测试、长度偏好或外部 verifier。后一类目标可能根本不可微，ZO/ES 此时不只是省显存，而是改变了可优化目标的范围。

## 2. Glossary

| 术语 | 本文中的含义 |
|---|---|
| FO / first-order | 使用反向传播得到梯度的训练。 |
| ZO / zeroth-order | 只查询函数值，借参数扰动估计更新方向。本文不把“training-free”自动等同于 ZO。 |
| two-point / antithetic | 同一方向测 `+ε` 和 `-ε`；通常比单边估计更稳，但需两次 forward。 |
| smoothing radius `σ` | 扰动幅度；太小会被数值/量化噪声淹没，太大会离开局部有效区域。 |
| population | 同一轮被评估的扰动候选数。 |
| fitness / reward | 候选参数的黑盒分数；不要求可微。 |
| ES | 用一群参数扰动的 fitness 加权更新参数或扰动分布。 |
| seed replay | 只保存随机种子，需要时重放随机数来恢复扰动。 |
| low-rank perturbation | 对矩阵权重不采完整高斯矩阵，而采形如 `ABᵀ` 的低秩扰动。 |
| subspace ZO | 只在全参数空间的一个低维子空间采样；子空间可以随机、敏感度驱动或 activation 驱动。 |
| rewarded perturbation | 在 support set / verifier 上得分高的参数扰动。 |
| recenter | 把下一轮采样中心从初始 `θ₀` 移到当前模型。 |
| quantized-space optimization | 直接更新离散低比特权重，或更新决定量化权重的连续 scale；两者应区别。 |
| gradient-free | 没有反向梯度。一个 pipeline 的某阶段 gradient-free，不代表蒸馏等后续阶段也无梯度。 |

## 3. 一张方法地图

| 工作 | 时间口径 | 扰动对象 | 选择/更新方式 | 代表规模 | 代码 | 定位 |
|---|---|---|---|---|---|---|
| [DiZO](https://arxiv.org/abs/2502.03304) | 2025，NeurIPS 2025 | LLM 参数 | 两点 ZO + 分层投影/尺度 | OPT/Llama | [代码](https://github.com/Skilteee/DiZO) | 核心 |
| [QuZO](https://arxiv.org/abs/2502.12346) | 2025，EMNLP 2025 | 低比特参数 | 两点 ZO + 优化随机舍入 | Llama2-7B | [代码](https://github.com/lloo099/QuZO) | 核心 |
| [ZO2 / DistZO2](https://arxiv.org/abs/2503.12668) | 2025 | 参数 | 标准 ZO + CPU/GPU offload/并行 | OPT-175B | [代码](https://github.com/liangyuwang/zo2) | 系统核心 |
| [QZO](https://arxiv.org/abs/2505.13430) | 2025，ICLR 2026 | 量化 scale | ZO + directional clipping | Llama2-13B、SD3.5 Large | [代码](https://github.com/maifoundations/QZO) | 核心/重参数化 |
| [ES at Scale](https://arxiv.org/abs/2509.24372) | 2025 | LLM 全参数 | population ES | 0.5B–8B | [代码](https://github.com/VsonicV/es-fine-tuning-paper) | 核心 |
| [FZOO](https://github.com/DKmiyan/FZOO) | ICLR 2026 | LLM 参数 | batched 单边估计 + 自适应步长 | OPT-66B、Llama3 | [代码](https://github.com/DKmiyan/FZOO) | 核心 |
| [ZO Fine-tuner](https://arxiv.org/abs/2510.00419) | 2025 | 参数组分布 | 学习可复用扰动策略 | 1B–30B | [代码](https://github.com/ASTRAL-Group/ZO_Fine_tuner) | 核心 |
| [EGGROLL](https://arxiv.org/abs/2511.16652) | 2025 首发，ICML 2026 | 低秩权重矩阵 | 大 population ES | RWKV 1.5B/7B | [代码](https://github.com/ESHyperscale/HyperscaleES) | 核心 |
| [QES](https://arxiv.org/abs/2602.03120) | 2026 | 离散量化全参数 | ES + error feedback | 1.5B/3B | [代码](https://github.com/dibbla/Quantized-Evolution-Strategies) | 核心 |
| [Neural Thickets（方法：RandOpt）](https://arxiv.org/abs/2603.12228) | 2026，ICML 2026 Spotlight | 预训练权重 | 独立采样、筛选、top-k ensemble | 0.5B–8B，扩展分析至 32B | [代码](https://github.com/sunrainyg/RandOpt) | 核心 |
| [CoRP](https://arxiv.org/abs/2605.31494) | 2026 preprint | rewarded perturbations | 兼容性加权、collapse、验证门 | 0.5B–8B | [代码](https://github.com/oooranz/CoRP) | 直接后续 |
| [SubZero](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.html) | 2024 预印本，ICCV 2025 | 低秩随机子空间 | 两点 ZO | OPT 等 | [代码](https://github.com/zimingyy/SubZero) | venue-window |
| [LOZO](https://github.com/optsuite/LOZO) | 2024 预印本，ICLR 2025 | 低秩矩阵 | 两点 ZO | autoregressive LM | [代码](https://github.com/optsuite/LOZO) | venue-window |
| [SensZOQ](https://openreview.net/forum?id=myYzr50xBh) | 2024 预印本，ICLR 2025 | 约 0.1% 敏感参数 | 稀疏 ZO | Llama2-7B | [代码](https://github.com/GarlGuo/SensZOQ) | venue-window |

“时间口径”很重要：若规则是**首次公开必须在 2025–2026**，最后三项应只作为背景；若规则是**正式发表在 2025–2026**，它们可以纳入正文。

## 4. 方法串讲（结合公开代码）

### 4.1 两点 ZO 的 2025–2026 改造

**DiZO：各层不应同幅度走。** [论文](https://arxiv.org/abs/2502.03304) 观察到 FO 与普通 ZO 的 layer-wise update pattern 不同，因此在标准正负扰动之上加入投影和按层适配，让不同层产生不同量级的更新。它仍是 MeZO 式 estimator，创新点是“更新之后如何分层整形”，而不是新的采样分布。论文覆盖 RoBERTa-large、OPT、Llama，并报告最多减少 48% GPU hours。仓库中的训练入口和 optimizer 逻辑位于 [DiZO](https://github.com/Skilteee/DiZO)；实现明显继承 Hugging Face/MeZO 风格，因此复现时要同时核对 projection 超参和基础 trainer 版本。

**FZOO：用 batch 统计把 ZO 变成吞吐友好的 workload。** [代码](https://github.com/DKmiyan/FZOO) 使用 Rademacher `±1` 扰动、batched one-sided estimates，并依据一批 loss 的标准差调节步长。其核心直觉是：若一次 forward 能并行评估多个轻量扰动，就减少逐方向 launch 和重复开销；归一化更新也缓和了尺度问题。官方 README 覆盖 RoBERTa-large、OPT 350M–66B、Phi-2 和 Llama3，并声称相对 MeZO 更少 forward passes。它牺牲了两点 antithetic 的对称性，收益依赖 batch 并行和自适应稳定化。

**ZO Fine-tuner：不再手工固定扰动分布。** [论文](https://arxiv.org/abs/2510.00419) 用 learning-to-learn 为一个 base model 训练紧凑 optimizer；每组参数共享少量扰动方差/策略参数，之后迁移到同一基座的下游任务和衍生 checkpoint。代码把“学习 optimizer”和“拿学到的 optimizer 做下游 ZO”拆成不同入口。优点是摊销调参成本，限制是前置 meta-training 本身带来成本，跨模型族迁移不能自然假定。

**SubZero / LOZO / SensZOQ：三种降维。** 这三项首次预印本在 2024，但正式 venue 在 2025，构成理解新作的必要背景：

- [SubZero](https://github.com/zimingyy/SubZero) 周期性构造/刷新随机低维子空间，并支持 full tuning、LoRA、prefix、prompt tuning；重点是降低估计维数。
- [LOZO](https://github.com/optsuite/LOZO) 直接把矩阵扰动写成低秩因子，匹配 LLM 微调梯度常呈低秩的经验；这也是后来 EGGROLL 的重要近邻，但两者的硬件目标和 population 规模不同。
- [SensZOQ](https://github.com/GarlGuo/SensZOQ) 先从源任务梯度得到可迁移的 static sensitive mask，目标阶段只对约 0.1% 参数做 ZO，其余权重可量化。严格说它需要一份来自源阶段的 FO 信息，目标微调才是 ZO。

### 4.2 系统与量化：让 forward-only 真正有用

**ZO2 / DistZO2。** [ZO2](https://arxiv.org/abs/2503.12668) 并未发明新的扰动 estimator，而是把两次 perturbed forward 与 transformer block 的 CPU/GPU 搬运重叠；其后 [DistZO2](https://arxiv.org/abs/2507.03211) 加入 perturbation parallelism、DDP 和二维并行。论文最醒目的结果是用 18GB GPU 加约 600GB CPU memory 微调 OPT-175B。正确解读是“GPU 容量门槛大幅下降”，而非总内存或总计算消失；本地仓库 [ZO2](https://github.com/liangyuwang/zo2) 的 README 也明确列出这项 CPU 内存要求。

**QuZO、QZO、QES 是三个不同层次。**

- [QuZO](https://arxiv.org/abs/2502.12346) 在 4/8-bit forward 中做 ZO，并优化 stochastic rounding 来压低量化引入的 bias；实验含 Llama2-7B。代码 [QuZO](https://github.com/lloo099/QuZO) 包含自定义 quantization kernel，环境较旧，复现风险高于普通 Hugging Face trainer。
- [QZO](https://arxiv.org/abs/2505.13430) 冻结低比特权重表示，主要扰动连续 quantization scale，并用 directional-derivative clipping 稳定估计；因此它是“重参数化空间 ZO”，不是直接给每个 int weight 加连续高斯噪声。实验覆盖 Llama2-13B 和 Stable Diffusion 3.5 Large，属于少数同时跨 LLM/大视觉生成模型的工作。
- [QES](https://arxiv.org/abs/2602.03120) 直接在离散量化全参数空间做 ES。量化步长会吞掉细小更新，于是它用 accumulated error feedback 保存未落到离散格点的高精度残差；seed replay 将候选模型的额外状态压到接近推理。仓库明确提供 INT4/INT8/W8A8 路径和 QuZO baseline。

### 4.3 ES at Scale：population 并没有被十亿维空间直接击垮

[Evolution Strategies at Scale](https://arxiv.org/abs/2509.24372) 是 RandOpt 最近的迭代式邻居。每轮从当前 LLM 全参数附近采样 population，运行 rollout 得到 reward，再将候选方向按 fitness 聚合；无需对 rollout 概率反传。实验从 0.5B 到 8B，包含 Qwen2.5-7B-Instruct，并与 PPO/GRPO 比较。公开仓库 [es-fine-tuning](https://github.com/VsonicV/es-fine-tuning-paper) 同时保留较早脚本和加速实现，README 明示代码仍在快速变化；复现应固定 commit，而不是只记论文超参。

2026 的 [The Blessing of Dimensionality in LLM Fine-tuning](https://arxiv.org/abs/2602.00170) 给出了很有用但不必过度数学化的解释：LLM 微调 reward landscape 的有效高曲率方向可能很少。随机向量虽然身处十亿维空间，许多向量仍共享足够的“有用方向”分量，所以 population 约 30 也可能找到改善候选。这是经验几何解释，不是对任意任务都成立的保证。

### 4.4 EGGROLL：低秩不是只为省参数，也是为提高 GPU 算术强度

[EGGROLL](https://arxiv.org/abs/2511.16652) 为每个矩阵权重采样 `A∈R^(m×r)` 与 `B∈R^(n×r)`，以 `ABᵀ` 替代完整随机矩阵。单个个体是低秩扰动，但 population 的加权和可以成为高秩更新。这样既把额外存储从 `mn` 降到 `r(m+n)`，也让大 population 的前向更适合现代加速器。论文报告 billion-parameter 大 population 下相对朴素 ES 的数量级加速，LLM 实验使用 RWKV 1.5B/7B 做 Countdown/GSM8K，并有纯整数 recurrent LM 预训练案例。

本地 [HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) 是 JAX research preview，官方建议先看 notebook 和 end-to-end test。它证明的是一种硬件可行路径，不等同于已经无缝支持所有 Transformer/vLLM 训练栈。

### 4.5 Neural Thickets / RandOpt：从“优化路径”转向“邻域里已有专家”

[Neural Thickets](https://arxiv.org/abs/2603.12228) 的核心实验不是反复估梯度，而是围绕预训练权重独立采样：

```text
θ₀ ──采样很多 seed 和 σ──> {θ₀ + σεᵢ}
   ──support set / verifier 打分──> top-k 专家
   ──推理时多数票或答案聚合──> 最终预测
```

代表设置是采 5000 个候选、保留 top 50。论文覆盖 Qwen、Llama、OLMo3 的 base/instruct 版本（0.5B–8B），任务包括 Countdown、GSM8K、MATH-500、OlympiadBench、MBPP、ROCStories、USPTO；还在 Qwen2.5-VL-3B-Instruct 上冻结视觉编码器、只扰动语言部分做 GQA。更大到 Qwen2.5-32B 的实验主要用于邻域密度分析。

它的关键发现不是“高斯噪声普遍增强模型”，而是：充分预训练后，附近可能密集存在不同任务专长的模型；单个随机专家的提升有限，多样性经 top-k aggregation 才释放。相应代价也很直接：大规模并行搜索，且 K=50 时每题约 50 份推理；依赖可评分 support set/verifier；从头训练或很弱模型不呈现同样现象。代码 [RandOpt](https://github.com/sunrainyg/RandOpt) 用随机 seed 重建逐参数噪声，并避免保存几千份完整模型。

仓库的 `iterative-randopt` 分支（已在本地 clone 中 fetch）是 2026-07 的重要扩展，但不是独立论文：每轮以约 30 个 perturbations 搜索、选 top 8，用 ensemble 产生正确轨迹/soft labels，再通过 SFT/KD 蒸馏为单模型并重新居中。搜索阶段无梯度，完整循环**不是** gradient-free，因为蒸馏使用反向传播。

### 4.6 CoRP：把一群 rewarded perturbations 压回一个模型

[CoRP](https://arxiv.org/abs/2605.31494) 直接回应 RandOpt 的两个痛点：top-k 推理成本和朴素均值会相互抵消。它对 rewarded candidates 做 reward weighting，再根据方向对齐与正交分散程度做 compatibility reweighting，合成候选更新，并用 held-out validation gate 决定是否接纳/重新居中。论文报告 useful variance 呈可复现低秩结构，但稳定均值方向只在部分情形出现，这解释了为何直接平均经常失败。

实验使用 Qwen2.5 0.5/1.5/3B、OLMo3-7B、Llama3.1-8B，以及与 RandOpt 相近的五类任务；population 约 500，最终只需一个模型 forward。它比 RandOpt 新很多，仓库 [CoRP](https://github.com/oooranz/CoRP) 目前也是初始 release，因此结果应视为有代码的早期证据，而非已经成熟的通用 recipe。

## 5. CV / 多模态：核心证据稀少，边界方法较多

检索得到的一个明确结论是：2025–2026 大视觉模型中，**真正随机扰动权重并替代反传，同时公开代码**的工作比 LLM 少。

- [Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](https://arxiv.org/abs/2506.12409)（AAAI 2026）是最贴题的无代码工作。它冻结 CLIP-ViT-B/16 主干，只训练 MoE adapter/LoRA，比较语言/视觉 branch 和不同层的 FO/ZO 混合；MoZO 对 ZO 方向做 sign normalization 并限制高方差视觉扰动。论文称省 89.1% memory，但代码仍未公开，故只列 roadmap。
- [RLR Optimizer](https://arxiv.org/abs/2502.00639)（ICLR 2026 oral）覆盖 Stable Diffusion 与图像/视频 diffusion alignment。它把 likelihood-ratio、ZO 和 FO estimator 递归组合为 “half-order”，仍利用模型可微性，不能称为纯参数扰动替代梯度。代码已 clone 至 [RLR-Optimizer](https://github.com/RTkenny/RLR-Optimizer)，适合作为相邻方法。
- [ZOO-Prune](https://arxiv.org/abs/2509.24837)（CVPR 2026）在 projection layer 注入高斯噪声，以输出变化估计 visual-token sensitivity，再做 training-free token pruning。它不更新模型权重，因此排除出核心；仓库 [ZOO-Prune](https://github.com/AIM-SKKU/ZOO-Prune) 仅用于边界核查。
- [Dual-Seed Evolutionary Algorithm](https://ojs.aaai.org/index.php/AAAI/article/view/37893) 用双目标 evolution 搜索 diffusion 初始 noise seed，而不是 parameter matrix；也是明确的排除项。
- [GS-ES](https://openreview.net/forum?id=McVjYBWMpT) 对 denoising action 做 ES-inspired sampling，但最终回传 fitness-weighted gradients；属于 hybrid roadmap。

## 6. 2026 roadmap（暂时无公开代码或尚不足以重点复现）

- [AGZO](https://arxiv.org/abs/2601.17261)：从 input activations 提取低秩基，在 activation-informed **weight-update subspace** 采样；activation 用来选参数方向，不是 activation noise。
- [AdaLeZO](https://arxiv.org/abs/2604.18264)：把“本步扰动哪一层”建模成 non-stationary bandit，以 replacement sampling 与 inverse-probability weighting 控制偏差。
- [ZO-Act](https://arxiv.org/abs/2607.01125)：初始化时从 activation 提取每层固定低秩 basis，冻结低比特原权重，只对系数矩阵做 ZO；刚发布，属重参数化空间扰动。
- [P-GAP](https://arxiv.org/abs/2510.18228)：先估低维 gradient space，再把随机扰动对齐到 projected direction；目前未找到官方仓库。
- [Overcoming Forgetting in LLM Fine-Tuning with Evolution Strategies](https://arxiv.org/abs/2605.30148)：用 anchored weight decay 把 ES 约束回初始模型附近，关注遗忘而非 estimator 吞吐。
- [RoZO](https://aclanthology.org/2026.eacl-long.80/)：在 LoRA adapter 空间做 geometry-aware ZO，规模和黑盒设定值得关注；当前未纳入代码重点。
- `Iterative RandOpt`：已有代码、尚无独立论文；因混合 SFT/KD，应跟踪为 RandOpt 工程路线而非纯 gradient-free 新算法。

## 7. 横向判断：该怎么选

| 你的约束 | 更自然的候选 | 主要代价/风险 |
|---|---|---|
| 单卡显存极紧、目标是常规 supervised loss | DiZO、FZOO、SubZero；超大模型看 ZO2 | 多 forward，收敛和超参可能弱于 Adam；ZO2 转移压力到 CPU/互联。 |
| reward 不可微、可以做 rollout | ES at Scale | population × rollout 的总推理成本；reward noise。 |
| 大 population 被 GPU 利用率卡住 | EGGROLL | 低秩结构与 JAX/模型栈限制；实现仍是 research preview。 |
| 模型已经 INT4/INT8 | QuZO / QZO / QES | 三者优化变量不同；量化格点会造成更新消失。 |
| 有大规模并行推理，想验证“邻域专家” | RandOpt | 搜索与 top-k inference 昂贵，且强依赖 verifier。 |
| 想把 RandOpt population 变成单模型 | CoRP，或 iterative RandOpt 蒸馏 | CoRP 很新；iterative pipeline 不再完全无梯度。 |
| 只可调 adapter/少数参数 | SensZOQ、SubZero、RoZO | 可能需要源任务梯度或额外 profiling。 |

## 8. 实验阅读与复现注意事项

1. **比较总成本，不只比较 peak GPU memory。** 至少记录 forward 次数、population、每候选 rollout 数、CPU RAM、互联、搜索与最终推理成本。
2. **把训练与 test-time ensemble 分开。** RandOpt K=50 的 accuracy 不能直接与单模型单 forward 当作相同部署预算比较；应同时报告 K=1、K=50 和蒸馏/CoRP。
3. **匹配目标函数。** supervised ZO 与 exact-match ES 优化的是不同对象；不可只凭最终 accuracy 断言 optimizer 更强。
4. **报告扰动空间。** 是全精度原权重、LoRA、低秩因子、quantization scale，还是离散 int 权重？“参数扰动”这个标签会掩盖巨大差异。
5. **固定 RNG 语义。** seed replay 对 PyTorch/CUDA 版本、参数遍历顺序和分布式 shard 非常敏感；checkpoint 应保存 seed、中心模型、`σ`、采样顺序和代码 commit。
6. **检查噪声是否淹没信号。** 小 `σ` 会遇到 fp16/int4 舍入与 reward stochasticity；大 `σ` 会离开预训练邻域。应扫 `σ` 并画正负候选的 score difference 分布。
7. **防止 verifier leakage / reward hacking。** support、selection、validation、test 必须拆开；格式正确带来的增益应与语义正确分开。
8. **多 seed 报告方差。** 这些方法显式依赖随机性，单次最好结果尤其容易乐观。

## 9. 本地代码清单

以下均已 clone 到本文同级目录；多数使用 shallow clone，RandOpt 额外 fetch 了 `iterative-randopt` 分支。它们是论文作者/项目公开仓库，不代表已下载模型、数据或安装依赖。

| 目录 | 当前用途 |
|---|---|
| [RandOpt](https://github.com/sunrainyg/RandOpt) | Neural Thickets 主实现与 iterative 分支 |
| [CoRP](https://github.com/oooranz/CoRP) | RandOpt population consolidation |
| [es-fine-tuning](https://github.com/VsonicV/es-fine-tuning-paper) | billion-parameter full ES |
| [HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) | EGGROLL / JAX 低秩 ES |
| [QES](https://github.com/dibbla/Quantized-Evolution-Strategies) | 离散量化空间 ES |
| [QZO](https://github.com/maifoundations/QZO)、[QuZO](https://github.com/lloo099/QuZO) | 两种不同量化 ZO 路线 |
| [DiZO](https://github.com/Skilteee/DiZO)、[FZOO](https://github.com/DKmiyan/FZOO)、[ZO-Fine-Tuner](https://github.com/ASTRAL-Group/ZO_Fine_tuner) | 2025–26 ZO optimizer 改造 |
| [ZO2](https://github.com/liangyuwang/zo2) | ZO offload / distributed system |
| [SubZero](https://github.com/zimingyy/SubZero)、[LOZO](https://github.com/optsuite/LOZO)、[SensZOQ](https://github.com/GarlGuo/SensZOQ)、[HiZOO](https://github.com/Yanjun-Zhao/HiZOO) | 2025 venue-window 与必要背景 |
| [RLR-Optimizer](https://github.com/RTkenny/RLR-Optimizer)、[ZOO-Prune](https://github.com/AIM-SKKU/ZOO-Prune) | CV/diffusion 相邻或排除边界 |

说明：`HiZOO` 的首次预印本为 2024、ICLR 2025 正式发表，故仅作为 Hessian-informed ZO 背景；它不是本 survey 的 2025 首发核心项。

## 10. 最后结论

这不是单一算法家族，而是同一接口——“采样扰动、只做 forward、按分数处理候选”——下的三种研究哲学。ZO 把扰动当梯度估计器，ES 把扰动当 population search，RandOpt 则把扰动当预训练邻域中已经存在的专家。2025–2026 的证据表明，大模型的有效微调几何可能远低于参数维数，低秩结构、按层结构和硬件批处理因此都能发挥作用；但它们尚未消除总 forward 成本、reward 设计、随机方差和部署时 ensemble 的负担。

当前最值得继续验证的两个问题是：一，RandOpt/CoRP 的“邻域专家密度”能否在没有强 verifier 的开放式任务上稳定出现；二，EGGROLL/QES 这类为 inference hardware 设计的扰动结构，能否在标准 Transformer 与主流 serving stack 上保持论文中的吞吐优势。它们决定这条路线会成为常规 post-training 工具，还是只在不可微目标、量化模型和超并行环境中的专用方法。
