# Qwen3-4B FFN `up / gate / down` permutation 验证方案

> 状态：**待审阅，尚未开始长程执行**  
> 日期：2026-07-10  
> 默认实验模型：本地 `models/Qwen3-4B`

## 1. 先给出预期结论

预期结论不是“`up`、`gate`、`down` 各自都可以随便 permutation”，而是：

- permutation 是 FFN **共享 intermediate-neuron 轴上的联合对称性**；
- `gate` 与 `up` 必须使用同一个行置换；
- `down` 必须在输入列上使用与之匹配的逆变换；
- 单独置换任意一个矩阵，或三个矩阵使用不一致的置换，通常都会改变 FFN 函数。

因此，本实验既要验证“正确的三矩阵联动置换确实成立”，也要用单矩阵和错误配对作为负对照，回答“feature 到底属于单矩阵，还是属于三矩阵共同定义的内部坐标系”。

## 2. 项目现状梳理

### 2.1 当前本地目标

- [`qwen3_infer.py`](../../scripts/qwen3_infer.py) 提供了本地模型位置与最小推理接口；新实验代码不会从中复制实验逻辑。
- 本地模型为 Qwen3-4B，权重约 7.6 GiB。
- 模型结构：36 层，`hidden_size=2560`，`intermediate_size=9728`，权重 dtype 为 BF16。
- Qwen3 的 MLP 为标准 SwiGLU：

  ```python
  down_proj(silu(gate_proj(x)) * up_proj(x))
  ```

- `gate_proj`、`up_proj`、`down_proj` 均为 `bias=False`，所以不需要额外处理 bias。
- 运行环境：conda 环境 `qwen3`，PyTorch 2.13.0+cu130，Transformers 4.57.6。
- 本轮模型加载、权重、hidden states、forward 和 logits 统一使用 **BF16**。
- 当前第二张 GPU 可用显存约 48 GiB；实验默认固定使用该卡，避免干扰第一张正在占用的 GPU。

### 2.2 实现隔离原则

本轮把 Qwen3-4B 视为独立实验对象，所有核心代码根据以下材料重新实现：

- Transformers 当前版本中的 `Qwen3MLP.forward`；
- 本地 Qwen3-4B 的 `config.json`；
- 本地 checkpoint 的真实 state-dict key、shape 和 dtype；
- 本文第 3 节给出的 SwiGLU permutation 关系。

本轮不从其他实验 import 或复制 permutation 实现，也不沿用其他实验阈值。新脚本必须从空文件开始实现，并用 Qwen3 专属单测独立验证；任何外部结果都不进入本轮的通过/失败判断。

## 3. 数学对象与待验证命题

令：

- 输入维度为 `d`，intermediate 维度为 `m`；
- `W_g, W_u ∈ R^(m×d)`；
- `W_d ∈ R^(d×m)`；
- `P ∈ R^(m×m)` 为 permutation matrix；
- `φ` 为逐元素 SiLU。

原始 FFN：

```text
g = φ(W_g x)
u = W_u x
h = g ⊙ u
y = W_d h
```

联动置换：

```text
W_g' = P W_g
W_u' = P W_u
W_d' = W_d P^T
```

因为逐元素运算满足：

```text
φ(Pz) = Pφ(z)
(Pa) ⊙ (Pb) = P(a ⊙ b)
P^T P = I
```

所以在精确实数运算下：

```text
y' = W_d P^T [φ(PW_gx) ⊙ (PW_ux)]
   = W_d P^T P [φ(W_gx) ⊙ (W_ux)]
   = y
```

PyTorch 中若 `perm` 的定义为 `z_perm = z[..., perm]`，对应实现是：

```python
gate.weight = gate.weight[perm, :]
up.weight   = up.weight[perm, :]
down.weight = down.weight[:, perm]
```

这里三处索引都写作 `perm`，但矩阵记号中 `down` 对应的是右乘 `P^T`。实验必须额外做方向性单测，避免把 `perm` 与 `argsort(perm)` 混淆。

## 4. 研究问题与假设

### 4.1 研究问题

1. Qwen3 SwiGLU 的 intermediate neuron 轴是否存在联动 permutation 对称性？
2. `up`、`gate`、`down` 中任一矩阵能否单独 permutation 而保持函数不变？
3. BF16 下的输出差异是置换公式错误，还是 `down` GEMM 归约顺序改变造成的有限精度误差？
4. BF16 数值误差经过 1 层与 36 层后如何传播？
5. 即使数学等价，工程上是否会明显改变 logits、top-1 token 或 greedy generation？

### 4.2 预注册假设

- **H1（联合对称性）**：同一个 `P` 联动作用于 `gate/up/down` 时，精确数学函数不变。
- **H2（单矩阵不成立）**：仅置换 `gate`、仅置换 `up` 或仅置换 `down`，函数通常改变。
- **H3（错误配对不成立）**：两个矩阵配对、三个独立置换、错误 inverse 方向均不能保持函数不变。
- **H4（误差来源）**：正确联动置换后，`gate/up/乘积` 可以与原结果逐坐标对齐；主要误差首次出现在 `down` 的浮点归约。
- **H5（BF16 分离度）**：有效置换在 BF16 下的数值漂移显著小于所有负对照。
- **H6（传播）**：有效置换的全模型误差可能随层传播并偶尔改变近似并列的 token，但这不否定 H1。

## 5. 实验分层

实验按 A → B → C 三层推进。每层通过后才进入下一层，避免全模型现象掩盖基础实现错误。

### A. 小矩阵代数与方向性单测

目的：不加载模型，先验证 permutation 定义、inverse 方向和判据。

设置：

- 小尺寸 `d=7, m=11`；
- 固定随机权重、固定输入；
- identity、单次 swap、reverse、随机 permutation；
- 所有张量和运算均使用 BF16；
- 每组至少 5 个 seed。

检查：

1. `perm` 确为 `[0, m)` 的双射，`inv_perm = argsort(perm)`。
2. 置换再 inverse 后，三个权重逐元素恢复，checksum 不变。
3. 显式检查：

   ```text
   g' == P g
   u' == P u
   h' == P h
   ```

4. 对比两种 `down` 路径：

   - native 路径：`down[:, perm] @ h'`；
   - canonical 路径：先把 `h'` inverse 回原顺序，再送入原 `down`。

   canonical 路径用于隔离浮点归约顺序；若它与 baseline 相同而 native 仅有微小误差，就能直接定位误差来源。

### B. Qwen3 单个真实 MLP 隔离实验

目的：在真实权重和真实 hidden state 上验证三矩阵关系，不让 Transformer 其余模块传播误差。

抽样层：

- layer 0；
- layer 17；
- layer 35。

输入：

- 固定 seed 的标准随机张量；
- 不同幅值的随机张量，用于观察数值尺度；
- 从固定 prompt forward hook 得到的真实 MLP 输入 hidden states。

精度统一为 BF16，与模型原生权重和后续全模型实验一致。

permutation 类型：

- identity；
- adjacent swap；
- reverse；
- 5 个固定随机 seed：`42, 43, 44, 45, 46`。

每个随机 permutation 运行下列对照：

| 组别 | gate | up | down | 预期 |
|---|---|---|---|---|
| baseline-repeat | 原 | 原 | 原 | bitwise 相同，验证确定性 |
| valid-triplet | `P` | `P` | matching `P^T` | 数学等价，仅有舍入误差 |
| gate-only | `P` | 原 | 原 | 不等价 |
| up-only | 原 | `P` | 原 | 不等价 |
| down-only | 原 | 原 | `P^T` | 不等价 |
| gate+up | `P` | `P` | 原 | 不等价 |
| gate+down | `P` | 原 | matching `P^T` | 不等价 |
| up+down | 原 | `P` | matching `P^T` | 不等价 |
| independent-triplet | `P_g` | `P_u` | `P_d^T` | 不等价 |
| wrong-direction | `P` | `P` | 错误 inverse 索引 | 不等价，用于抓方向 bug |

所有权重修改均在内存中完成；每个 case 后立即 inverse 恢复，并验证完整 checksum。不会生成额外的 7.6 GiB checkpoint。

### C. Qwen3-4B 全模型传播实验

目的：量化正确 permutation 在真实模型执行中的数值影响，而不是再次证明代数关系。

#### C1. 输入集

- 固定的中英文 prompt 集，覆盖常识、数学、翻译、代码、长短上下文；
- 约 32 条 prompt；
- 同一 tokenizer 结果落盘，后续所有 case 复用完全相同的 `input_ids/attention_mask`；
- forward 统计所有非 padding token，generation 只做补充观察。

#### C2. case

| case | 置换范围 | seed | dtype |
|---|---:|---:|---|
| baseline-repeat | 无，连续两次 | - | BF16 |
| one-layer-first | layer 0 | 42/43/44 | BF16 |
| one-layer-middle | layer 17 | 42/43/44 | BF16 |
| one-layer-last | layer 35 | 42/43/44 | BF16 |
| prefix-6 | layer 0–5 | 42/43/44 | BF16 |
| half-18 | layer 0–17 | 42/43/44 | BF16 |
| all-36 | layer 0–35 | 42/43/44 | BF16 |

负对照在全模型阶段只对 layer 17 做一次 `gate-only / up-only / down-only / independent-triplet`。单层阶段已经足以证明其不等价，无需用破坏后的模型做昂贵的全层实验。

#### C3. hook 与输出

- hook 每层 MLP 输入、MLP 输出和 decoder block 输出；
- 记录首个出现差异的位置及逐层误差增长；
- 记录 last-token 完整 logits；
- 对全部有效 token 在线计算聚合指标，避免落盘巨大的全量 logits；
- 对 baseline 与 `all-36` 运行确定性 greedy generation，`do_sample=False`；
- generation 是否文本一致只作为工程现象，不作为数学对称性的通过条件。

## 6. 指标

### 6.1 结构性指标

- `gate_coordinate_equal`：`g_perm` 与 `g_base[..., perm]` 是否相等；
- `up_coordinate_equal`；
- `product_coordinate_equal`；
- inverse restore 后每个权重的 `torch.equal` 与 SHA-256/checksum；
- native down 与 canonical down 的误差差异。

### 6.2 数值误差指标

对单层输出、hidden state 和 logits 均记录：

```text
max_abs      = max |a - b|
mean_abs     = mean |a - b|
rel_l2       = ||a - b||₂ / max(||a||₂, eps)
rel_linf     = ||a - b||∞ / max(||a||∞, eps)
cosine       = cosine_similarity(a, b)
```

logits 额外记录：

- top-1 agreement；
- top-5 set agreement；
- baseline top-1/top-2 margin；
- token 翻转是否集中发生在小 margin 样本；
- greedy completion exact-match。

### 6.3 与负对照的效应量

不能只说“有效置换误差不为零”，还需比较：

```text
separation_ratio = median(rel_l2 of negative controls)
                   / median(rel_l2 of valid-triplet)
```

它直接衡量“正常舍入误差”和“函数真的被改变”之间是否存在数量级分离。

## 7. 预注册判定标准

### 7.1 permutation feature 存在

同时满足以下条件，即确认 **FFN intermediate 轴存在联合 permutation symmetry**：

1. 公式与实现的方向性单测全部通过；
2. `gate/up/product` 在重排坐标后与 baseline 对齐；
3. canonical down 路径与 baseline 对齐；
4. valid-triplet 的 BF16 误差显著小于负对照；
5. 观察到的误差可以由 native down 与 canonical down 的差异定位；
6. 所有 case inverse restore 后权重与 baseline 完全一致。

注意：全模型 logits 是否 bitwise 相同**不是**此结论的必要条件。

### 7.2 单矩阵是否各自具有该 feature

- 若 `gate-only / up-only / down-only` 均稳定地产生远大于 valid-triplet 的误差，则结论为：

  > 三个矩阵单独都不具有任意 permutation 不变量；feature 属于共享 intermediate 坐标轴的联动重参数化。

- 默认要求负对照的 median `rel_l2` 至少为 valid-triplet 的 `100×`；若未达到，则扩大输入与 seed 后再判断，不能直接报“单矩阵也成立”。

### 7.3 工程数值稳定性分级

这部分与“数学 symmetry 是否存在”分开报告：

| 等级 | all-36 logits `rel_l2` | top-1 agreement | 含义 |
|---|---:|---:|---|
| 绿色 | `≤ 1e-3` | `≥ 99.9%` | 数值影响很小 |
| 黄色 | `≤ 2e-2` | `≥ 99%` | 存在可见漂移，通常仍可用于对齐研究 |
| 红色 | 超出黄色阈值 | `< 99%` | 需要把 kernel/精度敏感性纳入后续 merge 评估 |

该分级是工程风险标签，不会反过来否定精确实数域的代数结论。

## 8. 实现产物

审阅通过后，拟新增：

```text
experiments/ffn_permutation/
├── README.md                    # 复现入口
├── permutation.py               # apply / inverse / checksum
├── probe_synthetic.py           # A：BF16 小矩阵与方向性单测
├── probe_single_mlp.py          # B：真实 Qwen3 MLP 隔离实验
├── probe_full_model.py          # C：全模型实验
├── prompts.json                 # 固定输入集
├── run_all.sh                   # 分阶段、可恢复执行入口
├── results/
│   ├── manifest.json            # case 状态、耗时、环境信息
│   ├── synthetic.json
│   ├── single_mlp.jsonl
│   └── full_model.jsonl
├── logs/
└── RESULT.md                    # 最终结论、表格和异常说明
```

原始模型保持只读；新代码不依赖其他实验目录，也不落盘 permuted checkpoint。

## 9. 长程执行方式

审阅批准后的执行顺序：

1. 实现公共 permutation 与 restore 逻辑；
2. 运行 A，方向性或恢复检查失败则停止；
3. 运行 B 的一个 layer / 一个 seed smoke test；
4. 完成 B 的全矩阵；
5. 运行 C 的 BF16 全模型矩阵；
6. 汇总 JSON，生成 `RESULT.md`；
7. 对照预注册判据，分别给出“数学结论”和“工程数值结论”。

拟使用命令形式：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n qwen3 \
  python experiments/ffn_permutation/probe_synthetic.py

CUDA_VISIBLE_DEVICES=1 conda run -n qwen3 \
  python experiments/ffn_permutation/probe_single_mlp.py --resume

CUDA_VISIBLE_DEVICES=1 conda run -n qwen3 \
  python experiments/ffn_permutation/probe_full_model.py --resume
```

### 9.1 可恢复与安全约束

- 每个 case 完成后原子写入 manifest；中断后跳过已完成 case；
- 每次 permutation 后用 `try/finally` inverse restore；
- restore checksum 不一致立即停止，绝不继续使用污染模型；
- baseline 重复 forward 若不确定性超过测量误差，先定位环境/kernel，不进入结论阶段；
- GPU OOM 时优先降低 batch，不改变模型精度或实验判据；
- 所有实验固定 seed，关闭 sampling，记录 CUDA/PyTorch/Transformers 版本和 matmul 设置。

### 9.2 初步资源预算

- GPU：固定第二张 GPU；单卡执行；
- CPU 内存：当前可用约 206 GiB，足够模型加载、结果缓存与恢复校验；
- 磁盘：结果与日志预计低于 1 GiB；不复制模型权重；
- 预计墙钟时间：约 1–2 小时，实际以 smoke test 测得吞吐为准；
- 若全模型矩阵耗时显著超预算，会先汇报实测吞吐并征得确认，不会静默删减 case。

## 10. 最终报告应明确回答

最终 `RESULT.md` 必须分别回答：

1. 联动 permutation 在数学和实现层面是否成立？
2. `up`、`gate`、`down` 单独 permutation 是否成立？
3. 正确 permutation 的浮点误差首次出现在哪里？
4. BF16 误差量级与 seed、层位置、层数是什么关系？
5. Qwen3-4B 的 logits/token/generation 受影响到什么程度？
6. 该性质是否足以支持下一步“permutation 对齐后再 merge”，以及需要保留哪些数值风险提示？

## 11. 本轮不包含的范围

- 不做两个不同训练 checkpoint 之间的最优 assignment 搜索；
- 不做 permutation 后的权重 merge 或任务评测；
- 不修改或提交 7.6 GiB 模型权重；
- 不把 greedy 文本完全一致误当成函数等价的唯一证据。

这些应在本实验确认“对称性存在、实现正确”后，作为下一阶段单独设计。
