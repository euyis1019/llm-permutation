# Qwen3-4B FFN permutation 实验结论

> 执行日期：2026-07-10
> 预注册方案：[`ffn_permutation_experiment_plan.md`](../../docs/plans/ffn_permutation_experiment_plan.md)
> 环境：RTX 4090（48 GiB）单卡，PyTorch 2.13.0+cu130，Transformers 4.57.6，全程 BF16
> 复现：`bash run_all.sh`（约 10 分钟）；原始数据在 [`results/`](results/)

## 0. 一句话结论

**Qwen3-4B 的 SwiGLU FFN 在 intermediate-neuron 轴上存在联合 permutation 对称性：`gate`、`up` 用同一行置换、`down` 用匹配的列置换时，函数在精确实数域不变；BF16 下的全部残余误差被逐 bit 定位为 `down_proj` GEMM 的浮点归约顺序，且比任何单矩阵/错误配对置换小两个数量级以上。单独置换任一矩阵都会实质性改变函数。permutation 是三矩阵共享坐标系的性质，不属于任何单个矩阵。**

预注册判据 §7.1 的 6 项条件全部满足；§7.2 的 100× 分离要求实测 317×（最坏情况 235×）。

## 1. 执行概要与判据对照

| 阶段 | 内容 | 结果 |
|---|---|---|
| A | d=7, m=11 合成矩阵：方向性单测、inverse 恢复、fp64 判据校验（CPU+CUDA） | **通过，0 问题**（`results/synthetic.json`） |
| B | 真实 layer 0/17/35 MLP × 5 种输入 × 8 种 perm × 10 组对照，共 159 case | **通过**（`results/single_mlp.jsonl`） |
| C | 全模型 32 prompt × 23 case + greedy generation | **完成，无一致性异常**（`results/full_model*.jsonl`） |

§7.1 六项判定条件：

1. 公式/实现方向性单测全部通过（A 层；`wrong-direction` 陷阱在全部非对合 perm 上被抓住；fp64 下 valid-triplet `max_abs≈1.4e-14`，全部负对照 `rel_l2≈O(1)`）✅
2. `gate/up/product` 重排坐标后与 baseline **bitwise 相等**：B 层 120/120 ✅
3. canonical-down 路径与 baseline **bitwise 相等**：B 层 120/120（`rel_l2` 最大值恰为 0）✅
4. valid-triplet 的 BF16 误差显著小于负对照：单 MLP 分离度 **317×**（median），**235×**（最坏情况），超过预注册 100× 阈值 ✅
5. 误差可由 native/canonical down 差异定位（见 §3）✅
6. 全部 case inverse restore 后权重逐字节一致 + SHA-256 一致（B 159/159；C 每 case `torch.equal` vs CPU master copy + 全 MLP 权重全局 SHA 首尾一致）✅

## 2. §10 六个问题的回答

### Q1：联动 permutation 在数学和实现层面是否成立？——成立

- fp64 精确验证：`y' = W_d P^T [φ(PW_g x) ⊙ (PW_u x)] = y`，误差 ~1e-14（机器精度）。
- 真实 BF16 权重上：`g' == g[..., perm]`、`u' == u[..., perm]`、`h' == h[..., perm]` 在 3 个层 × 5 种输入 × 8 种 perm 全部 **bitwise 成立**——SiLU 和逐元素乘完全不引入误差。
- 实现约定已验证：`z_perm = z[..., perm]` ⇔ `gate/up: w[perm,:]`、`down: w[:,perm]`；inverse 用 `argsort(perm)`；方向写反立即被负对照抓出（单 MLP 输出 `rel_l2≈1.5`）。

### Q2：`up`、`gate`、`down` 单独置换是否成立？——全部不成立

单 MLP 输出 `rel_l2`（75 个 case/组：3 层 × 5 输入 × 5 seed）：

| 对照 | median rel_l2 | min | max |
|---|---:|---:|---:|
| baseline-repeat | **0（bitwise）** | 0 | 0 |
| **valid-triplet** | **4.72e-3** | 0 | 5.01e-3 |
| gate-only | 1.40 | 1.18 | 4.17 |
| up-only | 1.51 | 1.48 | 4.44 |
| down-only | 1.48 | 1.47 | 2.52 |
| gate+up | 1.48 | 1.47 | 2.48 |
| gate+down | 1.51 | 1.48 | 5.30 |
| up+down | 1.40 | 1.18 | 4.24 |
| independent-triplet | 1.54 | 1.44 | 3.56 |
| wrong-direction | 1.48 | 1.46 | 2.43 |

所有单矩阵、两两配对、三独立、方向写反的对照都把输出破坏到与 baseline 几乎不相关（`rel_l2 ≈ 1.4–1.5`，等价于随机向量间距离）。**feature 属于三矩阵共同定义的内部坐标系，不属于任何单个矩阵。**

### Q3：正确置换的浮点误差首次出现在哪里？——唯一出现在 `down_proj` GEMM 的归约顺序

三条互相独立的证据：

1. **canonical-down 隔离**：把置换后的 `h'` 用 `inv_perm` 恢复原序再送入原 `down_proj`，输出与 baseline **bitwise 相等**（120/120）。即置换路径上直到 `down` 输入为止零误差；native 路径（`down[:, perm] @ h'`）与之的差异就是全部误差。
2. **adjacent_swap 对照**：相邻两两交换只交换求和中相邻两项（浮点加法交换律精确成立、结合分组不变），BF16 输出 **bitwise 不变**（15/15）；而 reverse/random（改变结合分组）出现 ~5e-3 漂移。误差机制被精确锁定为**归约结合顺序**，而非置换公式。
3. **全模型 first-diff 定位**：23 个 case 中，首个非 bitwise 差异全部出现在**第一个被置换层的 `mlp_out`**（该层 `mlp_in` 仍 bitwise 相等），无一例外。

### Q4：BF16 误差量级与 seed、层位置、层数的关系

- **单层内与 seed/层位置几乎无关**：valid-triplet 单 MLP `rel_l2` 在 L0/L17/L35、5 个 seed 上均为 4.7–5.0e-3（非常稳定的 BF16 归约噪声水平）。
- **传播后饱和，不随层数线性累积**（全模型 logits，全 token `rel_l2` median）：

| case | logits rel_l2 (median, 3 seeds) | top-1 agreement |
|---|---:|---:|
| one-layer-last (L35) | 3.8e-3 | 99.4–99.5% |
| one-layer-middle (L17) | 1.7e-2 | 98.4–98.9% |
| one-layer-first (L0) | 2.1e-2 | 98.1–98.2% |
| prefix-6 (L0–5) | 2.1e-2 | 98.0–98.4% |
| half-18 (L0–17) | 2.2e-2 | 97.6–98.6% |
| all-36 (L0–35) | 2.0–2.2e-2 | 97.8–98.5% |

  越早注入放大越多（L0 单层 ≈ 全 36 层），说明主导因素是**扰动经过的深度**而非被置换层的数量；扰动在中后段被残差流"吸收"到 ~4e-3 的稳态相对水平，到最后几层与 lm_head 再放大到 ~1.4e-2（block_out 逐层曲线见 `results/summary.json` 的 `layerwise`）。

### Q5：logits / token / generation 受影响程度

- baseline-repeat：logits 与全部 108 个 hidden 流 **bitwise 确定**（32/32 prompt），排除环境非确定性。
- all-36 valid：全 token cosine ≥ 0.99933；top-1 翻转 19–28/1255 token（agreement 97.8–98.5%）。**翻转全部集中在近并列 token**：翻转位置的 baseline top1-top2 margin median 仅 0.03–0.125，而全体 token margin median 为 1.875。
- 最后一个 token 的 top-1：valid 各 case 为 32/32 一致（仅 all-36:s43 为 31/32）。
- greedy generation（64 tokens）：all-36 三个 seed 分别 27/32、26/32、29/32 与 baseline 完全一致；发散样例均为同义改写（如 "其数值为" → "它的数值是"），发生在开放式生成的近并列分叉处。
- 负对照（仅 L17 一层破坏）作为对比：logits `rel_l2≈0.30–0.34`，top-1 agreement 掉到 82–84%，last-token top-1 只有 59–81% 一致——与 valid 的 2e-2 / 98% 有清晰的数量级差别（logits 层面 ~18×，last-token ~23×）。

### Q6：是否足以支持"permutation 对齐后再 merge"？

**数学层面：可以。** 对称性存在、实现方向已验证、restore 机制可靠，可以放心把"同一 `P` 作用 gate/up、`P^T` 作用 down"作为后续对齐搜索的重参数化基元。

**工程数值风险提示（按 §7.3 预注册分级，如实报告为红色/临界）：**

- all-36 的 logits `rel_l2` median 2.0–2.2e-2、top-1 agreement 97.8–98.5%，**略超出黄色阈值（≤2e-2 且 ≥99%）**，按预注册表落入红色档。这不否定代数结论（判据 §7.1 明确 bitwise logits 非必要条件），但意味着：
  1. 下游 merge 评估必须把 BF16 kernel/归约敏感性纳入 ablation（例如 FP32 accumulate 的 down GEMM、或在 FP32 中做置换前后对比）；
  2. 以 greedy 文本 exact-match 作为 merge 质量指标会有 ~10–19% 的假阳性漂移，应改用 margin-aware 或 logits 级指标；
  3. 翻转集中在 margin < 0.13 的近并列 token，任务级指标（准确率类）受影响预计远小于 exact-match。

## 3. 执行偏离记录

1. **GPU 卡位**：计划固定第二张卡（GPU 1），但执行时 GPU 1 被另一用户 vLLM 进程占用 42 GiB（首次 Stage B 因此 OOM 一次，无数据污染），A 层已在 GPU 1 空闲时段完成，B/C 层改用完全空闲的 GPU 0。两卡同型号（RTX 4090），不影响结论。
2. **冒烟测试残留**：C 层 `--smoke`（4 prompt）曾写入正式结果文件，全量运行前已删除重建；`tokenized.json`（一次性 tokenize，含全部 32 prompt）保留复用。
3. 其余按计划执行：未落盘任何 permuted checkpoint；模型目录保持只读；每 case 均 try/finally inverse restore 并逐字节校验。

## 4. 原始数据索引

| 文件 | 内容 |
|---|---|
| `results/synthetic.json` | A 层全部检查与 fp64 对照表 |
| `results/single_mlp.jsonl` | B 层 159 case 逐条记录（坐标对齐、canonical-down、restore 校验） |
| `results/full_model.jsonl` | C 层 23 case × 32 prompt（逐层 rel_l2、bitwise 标志、logits/top-k/margin） |
| `results/full_model_generation.jsonl` | baseline 与 all-36 三 seed 的 greedy 生成及逐条 diff |
| `results/summary.json` | `summarize.py` 聚合输出（含逐层误差增长曲线） |
| `results/*manifest*.json` | case 状态、环境（matmul TF32=off，`float32_matmul_precision=highest`）、全 MLP 权重首尾 SHA |
