# Qwen3 4B 系列 FFN permutation benchmark 等价性与随机波动实验方案

> 状态：**待审阅，尚未执行**  
> 日期：2026-07-11  
> 上游实验：[`../ffn_permutation/RESULT.md`](../ffn_permutation/RESULT.md)、[`../ffn_permutation/BASE_RESULT.md`](../ffn_permutation/BASE_RESULT.md)  
> 模型：`Qwen3-4B`、`Qwen3-4B-Base`  
> 计算资源：`2 × NVIDIA GeForce RTX 4090`

## 1. 实验目的

已有实验已经证明：Qwen3-4B 的 SwiGLU FFN 在 intermediate-neuron 轴上存在严格的联合
permutation 对称性；BF16 部署时，正确置换仍会因为 `down_proj` GEMM 的有限精度行为产生
小幅 logits 漂移，并偶尔改变低 margin token。

本实验不再重复证明代数对称性，而是在 Qwen3-4B 与 Qwen3-4B-Base 上回答两个工程问题：

1. **阶段一——能力等价性确认：**正确置换全部 FFN 后，模型的 token 序列可以发生少量
   分叉，但最终答案和 benchmark correctness 是否仍在预注册容忍区间内等价？
2. **阶段二——随机 permutation 波动范围：**对大量独立随机 permutation，纯粹由参数
   排列改变 BF16 GEMM 数值路径所产生的 accuracy、答案和行为波动分布有多宽？

每次 permutation 都会产生权重布局不同、可独立加载的新 checkpoint，但在精确实数域中
仍表示同一函数；阶段二测量的是 **permutation-induced numerical variation**，不是训练噪声，
也不把这些 checkpoint 表述为获得了不同语义能力的新模型。

实验必须同时区分：

1. 生成文本是否逐 token 相同；
2. scorer 抽取出的最终答案是否相同；
3. benchmark 的对错与总体分数是否发生实质变化。

## 2. Preliminaries

### 2.1 已确认事实

- Qwen3-4B 共 36 层，FFN 为 `down(silu(gate(x)) * up(x))`；本地 Qwen3-4B-Base config
  也记录为 36 层、BF16、`Qwen3ForCausalLM`。执行前仍须逐 tensor 检查其 FFN 参数命名和
  联动置换规则，不能只凭模型名称或 config 假设。
- 对每层使用同一个 permutation：`gate/up` 置换行，`down` 置换匹配的列；不同层独立采样。
- 精确实数域中置换前后函数相同；当前实现已经通过方向、inverse 和负对照验证。
- Qwen3-4B all-36 BF16 实验的 logits `rel_l2` 约为 `2e-2`，全 token top-1 agreement
  约为 `98%`，翻转集中于 baseline margin 很小的位置。
- Qwen3-4B-Base 已按统一协议完成实验：all-36 logits `rel_l2` 约为 `1.5e-2`，全 token
  top-1 agreement 为 `98.0–98.3%`，三个 seed 的 64-token greedy exact match 均为 24/32。
- 漂移属于真实部署时的有限精度效应，因此保持 BF16，不使用 canonical-down 等仅适用于
  诊断的路径消除漂移。

### 2.2 模型与环境预检

预期原始 checkpoint 路径为：

```text
/nvme0/if/models/Qwen3-4B
/nvme0/if/models/Qwen3-4B-Base
```

执行前必须确认两个目录完整、可离线加载，并记录 config、tokenizer、权重文件的 SHA-256。
若路径不同，只能在冻结配置中显式修改；不得让 runner 自动从网络下载或静默换用缓存版本。

统一使用 conda 环境 `qwen3`。冻结并保存 Python、PyTorch、CUDA、vLLM、Transformers、
NCCL、GPU driver 版本和 `pip freeze`。模型间不得改变 kernel backend、dtype 或推理引擎版本。
当前已安装并通过单卡模型加载/generation smoke 的核心组合为 `vLLM 0.24.0`、
`PyTorch 2.11.0+cu130`、`Transformers 5.13.0`。由于该组合的 FlashInfer sampling JIT 与随包
CUDA headers 不兼容，conda 环境固定 `VLLM_USE_FLASHINFER_SAMPLER=0`；实验使用 greedy
decoding，不依赖 top-k/top-p sampler。该设置不关闭 FlashAttention、continuous batching 或
CUDA graph，并且必须在所有 baseline/permutation job 中保持一致。

### 2.3 评测基建

评测来源为 `/nvme0/if/llm-brewing/bench`。审阅通过后，将该目录完整硬拷贝到：

```text
experiments/ffn_benchmark_eval/bench/
```

后续执行只依赖本实验目录中的副本，不在运行时 import、软链接或读取原目录。复制后必须：

- 把 `benchmark_meta.json` 中的数据路径改为副本内的本地绝对路径；
- 检查 suite、代码和 meta 中不存在指向原仓或历史 `/mnt/...` 的有效依赖；
- 保存源目录与副本的文件清单和 SHA-256 manifest；
- 不修改评分协议、few-shot、prompt builder、stop tokens、标准答案或 EvalPlus tests。

## 3. 核心问题与假设

- **Q1：** checkpoint 重写或推理系统自身是否会造成可见差异？
- **Q2：** permutation 后有多少样本发生文本、最终答案或 correctness 分叉？
- **Q3：** 分叉是单向退化，还是低 margin 决策附近近似对称的 gain/loss？
- **Q4：** Qwen3-4B 与 Qwen3-4B-Base 是否给出一致结论？
- **Q5：** 三个确认 seed 是否满足预注册等价区间？
- **Q6：** 更多随机 seed 的均值、方差、分位数和 observed range 是多少，最差 seed 是否仍等价？

假设/预期：

- **H1：**固定软件、硬件和批处理配置后，baseline 重复运行及 baseline-copy 应完全一致；
  若不一致，先定位非确定性，不能用 `avg@k` 掩盖。
- **H2：**permutation 会使少量原始文本分叉，但最终答案与 correctness 的一致率更高。
- **H3：**baseline-only correct 与 permutation-only correct 数量近似平衡，不存在稳定单向退化。
- **H4：**两款模型的三个确认 seed 均与各自 baseline 实用等价。
- **H5：**阶段二多组随机置换结果的中心接近零，波动主要来自低 margin 样本，且不存在明显长尾退化。

## 4. 两阶段模型实验组

所有实验都从对应模型自己的原始 checkpoint 生成，不跨模型共享 permutation 后权重。

### 4.1 阶段一：能力等价性确认

对 `Qwen3-4B` 和 `Qwen3-4B-Base` 分别生成：

| 模型标签 | 处理 | 用途 |
| --- | --- | --- |
| `<family>__baseline_original` | 原始模型，只读 | 主基线 |
| `<family>__baseline_copy` | 不置换，按相同流程重新保存 | checkpoint 重写对照 |
| `<family>__perm_all36_s42` | 36 层正确联动置换，seed 42 | 确认 seed |
| `<family>__perm_all36_s43` | 36 层正确联动置换，seed 43 | 确认 seed |
| `<family>__perm_all36_s44` | 36 层正确联动置换，seed 44 | 确认 seed |

### 4.2 阶段二：多组随机置换评测

阶段一通过后，对每个模型族额外生成 20 组 all-36 随机置换（由 20 个不同随机种子各生成一套置换及对应 checkpoint）。随机种子在
执行前一次性写入冻结配置，建议固定为 `1000..1019`；不得根据阶段一或中途结果换 seed、
剔除 seed 或只保留表现较好的 checkpoint。

阶段二的目标是估计随机 permutation 的波动分布，因此阶段一的三个确认 seed 不计入这 20 组
随机置换样本。若未来增加随机置换组数，必须在查看追加结果前记录追加规则和数量。

### 4.3 checkpoint 验证与生命周期

每个 seed 在各层独立生成 permutation，算法与已有 `ffn_permutation` 实验保持一致。每个
checkpoint 写入 manifest，至少记录模型族、源模型、层数、seed、PRNG 算法、每层
permutation hash、权重 dtype、文件 hash 和生成脚本版本。生成后验证：

- 非 FFN 权重与源模型逐 tensor 相同；
- FFN 权重满足预期的行/列置换关系；
- `baseline_copy` 的全部 tensor 与源模型相同；
- checkpoint 能由冻结的 vLLM 环境加载并完成固定 prompt 的 smoke generation。

阶段二允许采用 rolling checkpoint：CPU 生成一个 seed、完成验证和评测后再生成下一个，
避免同时保存 40 份约 4B 参数权重。只有在 raw result、manifest、hash 和生成日志均已安全
落盘后才能删除临时权重；原始模型、阶段一 checkpoint 和所有 manifest 必须保留。

## 5. Benchmark 与执行阶段

### 5.1 固定 benchmark

主实验只使用以下六项：

```text
mmlu, gsm8k, ceval, cmmlu, humaneval_plus, mbpp_plus
```

- MMLU：英文通用知识；
- GSM8K：数学推理与答案生成；
- C-Eval、CMMLU：中文知识；
- HumanEval+、MBPP+：可执行代码生成，使用 EvalPlus external runner。

`external` 是 runner 类型而不是 benchmark 名。删除 `mmlu_redux`、`mmlu_pro`、`bbh`、
`math500` 和 `cruxeval`，代码评测不再作为可选补充阶段。

四个 protocol benchmark 各使用预先固定的 500 条样本；MMLU、C-Eval、CMMLU 必须按
subject 确定性分层抽样，GSM8K 使用固定 sample ID，不得简单依赖可能变化的文件顺序。
HumanEval+（164）和 MBPP+（378）使用全量。两个模型族及全部 checkpoint 使用完全相同的
sample IDs、prompts、tests 和 generation 参数。

### 5.2 Smoke 与吞吐校准

先用两个 baseline、两个 baseline-copy 和每个模型族的 `s42`，在六种协议上各取少量固定
样本，验证加载、落盘、sample ID 对齐、scorer/EvalPlus、断点续跑和 baseline 确定性。

正式评测前允许在 **baseline smoke prompts** 上做一次只面向性能的短校准。校准不得查看或
比较 benchmark correctness，只比较 tokens/s、GPU utilization、显存峰值、OOM 和稳定性。
建议测试：

- `max_num_seqs ∈ {128, 256, 512}`；
- `max_num_batched_tokens ∈ {16384, 32768, 65536}`；
- `gpu_memory_utilization` 从 `0.90` 起，在无 OOM 前提下最多提高到 `0.95`；
- `max_model_len` 设为覆盖冻结 prompt + generation 上限的最小安全值，不盲目使用模型最大值；
- BF16、prefix caching 开启；CUDA graph 保持开启，只有确认不兼容时才使用 eager mode。

选择“无 OOM 且吞吐最高”的一组参数后立即冻结；两款模型如果因 config 不同需要不同参数，
分别冻结，但同一模型族的 baseline 与全部 permutation checkpoint 必须完全一致。

Smoke 和吞吐校准通过前不得提交完整矩阵。

### 5.3 阶段一 Main

两款模型分别运行 `baseline_original`、`baseline_copy`、三个确认 seed 和六个 benchmark。
baseline_original 以完全相同配置独立运行两次，用于确定性检查；主比较只使用第一份有效
baseline 输出。greedy repeat 不是统计重复，统一 `n_runs=1`，不计算 `avg@k`。

### 5.4 阶段二多组随机置换评测

阶段一满足有效性前提后，两款模型分别运行额外 20 组随机置换。使用与阶段一完全相同的六项
benchmark、样本、prompt、scorer、EvalPlus tests、GPU 配置和 batch 参数。阶段二不重新运行
baseline，直接复用阶段一冻结的有效 baseline raw results。

## 6. 2×RTX 4090 并行、并发与 vLLM 策略

高 GPU 利用率是实验协议的一部分，不是可选优化。执行脚本必须同时提供 **跨 GPU 并行**和
**单 GPU 内 continuous batching 并发**，禁止逐 prompt 串行请求。

### 6.1 拓扑与任务调度

- 默认 `tensor_parallel_size=1`：4B BF16 checkpoint 可放入单张 4090，两个独立 worker
  分别绑定 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，同时处理两个不同任务。
- 不默认使用 TP=2。两张 4090 之间的 tensor-parallel 通信会占用带宽，并把可同时运行的
  worker 数从 2 降为 1；只有 smoke 实测 TP=2 吞吐更高时才允许改动，并记录证据。
- 全局 scheduler 维持最多两个 GPU worker，每张卡同一时刻恰好一个 vLLM engine；不得在
  同一张卡上重叠加载两个 checkpoint，也不得让两个任务误用同一 GPU。
- 调度单位为 `(model_family, checkpoint, benchmark)`，使用动态工作队列而非静态地把一半
  任务永久分给某张卡，避免 HumanEval+/MBPP+ 或长输出任务造成尾部空闲。
- 两个 worker 同时运行不同 checkpoint 或 benchmark；同一 paired comparison 不要求同一
  时刻运行，但必须使用冻结配置。任务完成即领取下一个，直到队列清空。
- checkpoint 生成、hash、结果分析和 EvalPlus CPU sandbox scoring 与 GPU generation 做流水线
  并发，但设置 CPU/IO 并发上限，避免抢占导致 GPU 等待或磁盘抖动。

### 6.2 无 client 的离线批量推理

- 禁止启动 HTTP/OpenAI-compatible server，禁止 deploy/client 拆分，也不使用 client-side
  concurrency 参数。
- protocol benchmark 在 worker 内只加载一次模型；同一 checkpoint 的 MMLU、GSM8K、
  C-Eval、CMMLU 尽量在同一进程依次完成，避免为每个 benchmark 重复加载权重。
- 每个 benchmark 先构建全部 prompts，再一次性调用 `llm.generate(all_prompts, params)`；
  “一次性提交”不要求全部 sequence 同时常驻显存，vLLM 根据 KV cache 自动 continuous batch，
  在无 OOM 的前提下保持 GPU 饱和。
- HumanEval+/MBPP+ 使用 EvalPlus 的 in-process vLLM provider，同样一次提交任务集合；代码执行
  和 tests 在 generation 后并发评分，不经过网络 client。
- 固定 `temperature=0`、BF16 和每项 benchmark 的 stop/max-token 协议。`n_runs=1`；不得通过
  复制相同 prompts 制造虚假的并发或统计重复。

### 6.3 利用率监控与回退

每个 job 记录 wall time、prompts/s、input/output tokens/s、GPU utilization、显存峰值、队列
等待时间和失败重试。若稳定阶段任一 GPU 连续低利用率，先检查数据准备、CPU scoring、磁盘
读取和 batch 参数；不得以启动 HTTP client 作为修复手段。OOM 时按冻结顺序回退
`max_num_seqs`、`max_num_batched_tokens`、`gpu_memory_utilization`，并对所有同族 checkpoint
统一重跑受影响任务，不能只给某个 permutation seed 使用更保守配置。

## 7. 指标与统计

每道题必须保存 baseline 与各实验 checkpoint 的配对结果，至少包含：

- sample ID、原始 response、scorer 抽取答案和 correctness/pass；
- 原始文本 exact match；
- 抽取答案 agreement；
- correctness agreement；
- baseline correct / permutation wrong 数量（loss）；
- baseline wrong / permutation correct 数量（gain）。

EvalPlus 结果必须归一化为 sample-level response 和 pass/fail 后再进入 paired 分析；只有总体
pass@1 而无逐题结果时，不得声称完成代码 benchmark 的 gain/loss、McNemar 或 paired bootstrap。

每个 benchmark、每个 permutation seed 分别报告：

```text
accuracy_delta = accuracy_perm - accuracy_baseline
behavior_disagreement = P(correct_perm != correct_baseline)
net_change = (gain - loss) / N
```

阶段一使用题目级 paired bootstrap 95% CI，并报告 McNemar exact test。六任务平均分先在
每个 benchmark 内算 delta，再对六项 benchmark 等权平均，不把所有题直接混成一个大样本。

阶段二对每个模型族、每个 benchmark 以及六任务平均分分别报告这 20 组随机置换的：mean、standard
deviation、median、IQR、5%/95% quantile、observed min/max range；同时画 seed-level delta、
disagreement 和 gain/loss 分布。`min/max` 只称为“20 组预注册随机置换的观测范围”，不外推为所有
可能 permutation 的理论上下界。

“没有显著差异”不能作为等价结论。预注册实用等价区间为：

- 单 benchmark：`[-1.0, +1.0]` percentage point；
- 六任务平均分：`[-0.5, +0.5]` percentage point。

## 8. 判定规则

1. **评测有效性前提：**每个模型族的两次 baseline_original 与 baseline_copy 的 response、
   抽取答案、correctness 必须完全一致。若不一致，停止能力等价判断并定位 checkpoint、batch、
   GPU 或 kernel 非确定性。
2. **阶段一强等价：**两款模型的三个确认 seed 在每个 benchmark 上 paired 95% CI 均完整
   落入 `±1.0 pp`，且各 seed 的六任务平均分 paired 95% CI 均完整落入 `±0.5 pp`。
3. **阶段一六任务整体等价：**六任务平均分 CI 满足 `±0.5 pp`，且没有同一 benchmark 在至少
   两个确认 seed 上出现超过 `1.0 pp` 的稳定退化，但个别 benchmark CI 因样本量过宽。
4. **确认存在实质退化：**任一模型族中，任一 benchmark 在至少两个确认 seed 上的 paired
   95% CI 上界均低于 `-1.0 pp`，或任一确认 seed 的六任务平均分 CI 上界低于 `-0.5 pp`。
5. **阶段二随机波动结论：**分别陈述两款模型的中心、离散度、尾部和 observed range，并报告
   20 组随机置换中越过等价界限的数量；阶段二不因少量极端置换事后修改阶段一判定规则。
6. 若 CI 仅因样本量不足而无法落入等价区间，结论写作“证据不足”；只能按预注册规则追加
   全量数据，不能临时放宽阈值。

文本 exact match 下降本身不构成能力退化。最终报告必须并列展示文本、答案和 correctness
三层结果，并分别报告 Qwen3-4B 与 Qwen3-4B-Base，不能只给合并平均数。

## 9. 预期产物

```text
experiments/ffn_benchmark_eval/
├── README.md
├── bench/                         # 评测基建与数据的硬拷贝
├── configs/                       # 冻结环境、模型、seed、样本与 vLLM 配置
├── scripts/
│   ├── prepare_models.py
│   ├── validate_models.py
│   ├── calibrate_vllm.py
│   ├── run_smoke.sh
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── gpu_worker.py
│   └── analyze_paired.py
├── model_manifests/
├── results/
│   ├── raw/
│   ├── paired/
│   ├── stage1_summary.json
│   ├── stage2_distribution.json
│   └── figures/
├── logs/
│   ├── environment/
│   ├── throughput/
│   └── jobs/
└── RESULT.md
```

所有执行必须支持断点续跑、任务级文件锁和 atomic rename；已有结果不得静默覆盖。失败重跑
必须记录原因、attempt、输入 hash 和输出路径。`RESULT.md` 最终回答：置换造成多少文本、答案
与 correctness 分叉，gain/loss 是否平衡，两款模型是否一致，阶段一支持哪一级等价结论，以及
阶段二 20 个随机 permutation 的数值波动范围有多宽。

## 10. 本轮不包含

- 不重新证明 FFN permutation 的数学正确性或误差来源；
- 不搜索最优 permutation，不按 benchmark 结果选择 seed，不做模型 merge；
- 不使用随机采样、temperature sweep 或 pass@k；
- 不把 greedy repeat 当作独立样本，不使用 `avg@k` 掩盖非确定性；
- 不使用 HTTP client/server，不逐 prompt 串行推理；
- 不在主结果出来后按结果选择、删除或替换 benchmark。
