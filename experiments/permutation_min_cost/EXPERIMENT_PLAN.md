# 最小代价置换定律实验（permutation_min_cost）

> 状态：**已预注册，待执行**
> 日期：2026-07-11
> 模型：`Qwen3-4B-Base`（主线全部阶段）；`Qwen3-4B`（仅阶段四 3 个点的方向确认）
> 上游实验：
> - 代数/机制层：[`../ffn_permutation/RESULT.md`](../ffn_permutation/RESULT.md)
> - benchmark 层：[`../ffn_benchmark_eval/RESULT.md`](../ffn_benchmark_eval/RESULT.md)
> 计算资源：**仅允许使用 GPU 0，且必须与其他用户共存**（见 §8）

---

## 0. 给执行 Agent 的总指令（必读）

1. **全程自主执行，中途禁止向用户寻求确认、提问或征求偏好。** 本文档已预注册全部设计决策；文档没写的实现细节由你自行做合理选择并记录在 `DECISIONS.md`，但**不得改变**：置换族定义、几何指标集合、验收阈值、seed、模型、benchmark 集合、判定规则。
2. **分阶段推进，每阶段先跑完、写完该阶段的验收核对（§9），达标才进入下一阶段。**
3. **任一阶段未达验收标准：立即停止，不进入后续阶段，不修改阈值重跑，不换 seed 重试。** 写出 `FAILURE_REPORT.md`（§10 给出模板），然后结束整个任务。未达标本身是有效的科学结论，按 §10 如实报告即可。
4. **基础设施类失败**（GPU 长期不可用、反复 OOM 超过重试计划、磁盘不足）：按 §10.2 写 `STATUS_REPORT.md` 后停止。**不要**为了继续跑而突破 §8 的资源约束。
5. 所有执行必须可断点续跑（结果文件存在且完整即跳过）、原子写入、不静默覆盖已有结果。
6. 完成全部阶段后写 `RESULT.md`（§11 给出必答问题清单），任务结束。

---

## 1. 背景：已确立的事实（不要重复验证）

以下结论来自上游两个实验，本实验直接以其为前提：

1. SwiGLU FFN 对中间维联动置换（`gate/up` 行置换、`down` 列置换同一 `P`）在实数域**严格等价**；实现约定与校验工具在 `scripts/permutation.py`（已从上游复制）。
2. BF16 下置换的**全部**残余漂移逐比特定位于 `down_proj` GEMM 的浮点归约顺序；`gate/up/silu/mul` 在置换下逐比特不变。相邻交换（adjacent swap）单层输出逐比特不变；reverse/random 产生 ~5e-3 的单层 `rel_l2`。
3. 全模型层面：漂移主导因素是**扰动传播深度**（只置换 L0 ≈ 全 36 层的效果；只置换 L35 几乎无影响），中后段饱和。
4. benchmark 层面（Qwen3-4B-Base，20 组全局随机 all-36 置换，由 20 个不同随机种子各生成一套置换）：多选题（MMLU/C-Eval/CMMLU）几乎免疫（0/20 组越过 ±1pp）；生成类（GSM8K/HumanEval+/MBPP+）有小而真实的偏移；六任务平均分 +0.49±0.21pp。**这 20 组随机置换形成的分布是本实验阶段三的参照系，原始数据在 `../ffn_benchmark_eval/results/`，不要重跑。**
5. 推理确定性：在本机 GPU 0 + `VLLM_BATCH_INVARIANT=1` + `enforce_eager=True` + 关闭 prefix caching 下，Qwen3-4B-Base 的 greedy 评测**逐 run 完全确定**（11 个同权重 baseline 两两 correctness 零差异）。因此 **avg@k 无意义，一律 n_runs=1**；统计量只来自"同一策略换置换 seed"。
6. 结果对 `gpu_memory_utilization` 不变（0.28 与 0.90 correctness 零差异），两种设置的结果可混用。

## 2. 实验目标（一句话）

**建立一条可提前计算的定律：给定一个置换的几何结构（动了多少神经元 K、挪了多远 D、多局部 W），预测它在 BF16 部署下造成的漂移；并用该定律回答"必须置换 K% 神经元时怎么摆代价最小"，在 benchmark 上验证。**

"好/坏"的两层定义（预注册）：

- **logit 层**：与原模型 logits 的相似度——全模型 forward 的 `rel_l2`（median over prompts）、全 token top-1 一致率、低-margin 翻转数。
- **benchmark 层**：6 项 benchmark 的配对 accuracy delta 与六任务平均分变化（协议与 `../ffn_benchmark_eval` 完全一致）。

## 3. 明确排除（禁止做）

- ❌ Attention / 非 FFN 矩阵的置换；
- ❌ 数据感知的神经元选择（按激活/贡献挑神经元）——仅当阶段一失败时在 FAILURE_REPORT 中**建议**它，不执行；
- ❌ Qwen3.5-9B 或任何其他模型；
- ❌ 多 kernel 对比实验（但代码必须留接口，文档必须写 caveat，见 §7）；
- ❌ avg@k / 采样解码 / temperature>0；
- ❌ 在 benchmark 上搜索策略（benchmark 只做最终验证，候选筛选全部在单层与 logit 层完成）；
- ❌ 杀掉或干扰其他用户的 GPU 进程；使用 GPU 1。

## 4. 置换族（预注册，实现于 `scripts/perm_families.py`）

记 `m = 9728`（Qwen3-4B intermediate size）。所有族生成 `[0,m)` 的双射，用 `permutation.check_bijection` 校验。每个配置在 36 层用独立 seed：`layer_seed = base_seed*1000 + layer_idx`（与上游一致）。

| 族 | 名称 | 构造 | 扫描参数 | 目的 |
|---|---|---|---|---|
| F1 | window-shuffle | 把 `[0,m)` 切成宽 W 的连续窗口，窗口内随机洗牌 | W ∈ {2, 8, 32, 128, 512, 2048, 9728} | 局部性主轴（K≈100% 但位移 ≤W） |
| F2 | K%-block-local | 随机选一段长 ⌊K·m⌋ 的连续块，仅块内随机洗牌 | K ∈ {5%, 10%, 30%, 50%} | 局部大块 |
| F3 | K%-scattered-global | 随机选 ⌊K·m⌋ 个散布 index，在它们之间做随机错排 | K ∈ {5%, 10%, 30%, 50%} | 朴素"随机撒 K%"基线 |
| F4 | K%-adjacent-pairs | 随机选 ⌊K·m/2⌋ 个互不相交的相邻对 (i,i+1)，逐对交换 | K ∈ {10%, 30%, 50%, 100%} | 固定 K 下位移最小 |
| F5 | K%-strided-pairs | 固定 K=30%：选 ⌊K·m/2⌋ 对 (i, i+D)，逐对交换（对不相交） | D ∈ {1, 4, 16, 64, 256, 1024, 4096} | **关键轴：K 恒定、只变位移距离** |
| F6 | block-swap | 交换两个长 ⌊K·m/2⌋ 的连续块，块起点间距 D=m/2；K ∈ {10%, 30%} | K 两档 | 大而整体的位移 |
| F7 | global-random | 全局随机置换（= 上游主实验） | — | 上界参照 |
| F8 | reverse / identity | 整体倒序 / 恒等 | — | 参照与阴性对照 |

seed：每个配置 5 个 seed，**固定为 {201, 202, 203, 204, 205}**（F8 无 seed）。任何情况下不得增删换 seed。

## 5. 几何指标（预注册，实现于 `scripts/geometry.py`）

对每个置换 P 计算（moved = {i : P(i)≠i}）：

1. `frac_moved` = |moved|/m
2. `mean_disp` = mean_{i∈moved} |P(i)−i|
3. `max_disp` = max |P(i)−i|
4. `total_disp` = Σ|P(i)−i| / m
5. `cross_B` (B ∈ {16, 64, 256})：Σ_i ⌈|把 i 从块⌊i/B⌋ 挪到块⌊P(i)/B⌋是否跨块⌉ / m（近似归约块跨越数）
6. `inversions`：逆序对数的蒙特卡洛估计（采样 10⁶ 对）/ 归一化

回归做法：对每个指标 g，在**非逐比特相等**的配置上拟合 `log(drift) ~ log(g)`，报告 R² 与 Spearman ρ；同时报告二元组合 `(frac_moved × mean_disp)` 与 `cross_64` 的表现。**指标集合与回归形式不得中途增删**（探索性附加分析可以放附录，但验收只看上述预注册集合）。

## 6. 分层设计

### 阶段一：单层定律（成本：分钟级，GPU 显存 <2GB）

- **对象**：Qwen3-4B-Base 的 L0 / L17 / L35 三层真实 `down_proj` 权重。
- **输入**：(a) 真实激活——用与上游相同的固定 prompt（`../ffn_permutation/prompts.json`）hook 出各层 MLP 输入 x，计算 `h = silu(gate(x))·up(x)`（BF16，置换下逐比特不变，可放心复用）；(b) 合成输入 randn seed 7 的 ×1 与 ×10 两个尺度（鲁棒性检查）。
- **测量**：`y_perm = matmul(W_d[:,P], h[P])` vs `y = matmul(W_d, h)`，BF16，GPU 0。漂移 = fp32 下的 `rel_l2`（用 `permutation.diff_metrics`）。
- **matmul 后端**：统一经由 `scripts/backend.py` 的 `matmul(a, b, backend="torch_bf16")` 单点调用——这是 kernel 切换预留接口，本实验只用默认后端。
- **规模**：3 层 × 3 输入 × (F1:7 + F2:4 + F3:4 + F4:4 + F5:7 + F6:2 + F7:1 + F8:2 =31 配置) × 5 seed ≈ 4200 次测量（F8 除外）。
- **产出**：`results/stage1_singlelayer.jsonl`（每行一个测量：族/参数/seed/层/输入/几何指标全集/漂移指标全集）+ `results/stage1_regression.json`（每个几何指标的 R²、ρ）+ 曲线图。

### 阶段二：定律穿透到全模型 logits（成本：每配置约 1–2 分钟 forward，显存 ~10GB）

- **候选**：从阶段一结果中选 **24 个配置**覆盖漂移量程：预测漂移最小 8 个（含 F4-K30、F5-D1）、中间 8 个、最大 8 个（含 F7、F6）。每配置 1 个 seed（固定 201）。**选择规则如上，不得按全模型结果回头换候选。**
- **做法**：内存中对全部 36 层施加置换（同族同参数、逐层独立 seed），复用上游 `probe_full_model.py` 的协议：32 个固定 prompt，一次 forward，测 logits `rel_l2`（median）、全 token top-1 一致率、低-margin 翻转数；然后逆置换恢复并逐字节校验（`try/finally` + SHA 校验，上游已有实现模式）。**不落盘 checkpoint。**
- **产出**：`results/stage2_fullmodel.jsonl` + 阶段一预测值 vs 全模型实测的散点与 Spearman ρ。

### 阶段三：benchmark 终验（成本：每 checkpoint ≈ 4–8 分钟，共 9 个）

- **三个策略臂，全部 K=30%**（按阶段一定律确定构造细节，但臂的语义固定）：
  - **min-cost**：定律给出的最小代价摆法（预期为 F4-adjacent-pairs 或 F5-D1，以阶段一实测为准）；
  - **naive**：F3-scattered-global（朴素随机撒 30%）；
  - **worst**：定律给出的最大代价摆法（预期为 F6 或 F3 中 cross_B 最大者）。
- 每臂 3 个 seed {201, 202, 203} → 9 个 checkpoint，Qwen3-4B-Base。
- **完全复用** `../ffn_benchmark_eval` 的基建：`make_checkpoint.py` 模式生成落盘 checkpoint（需给 `make_checkpoint.py` 增加按本实验族构造置换的入口，或在本实验 scripts 里写等价生成器 + 同等验证：非 FFN 张量逐比特不变、FFN 满足行/列置换关系）、`run_worker.py` 跑 6 benchmark、配对分析逻辑。baseline 用**已冻结的** `../ffn_benchmark_eval/results/raw/qwen3_4b_base__baseline_original_run1/`，不重跑。
- 评测配置**必须**与上游完全一致：GPU 0、`VLLM_BATCH_INVARIANT=1`、eager、无 prefix caching、`max_model_len=4096`、greedy、n_runs=1、`gpu_memory_utilization=0.28`（若 GPU 空闲可用 0.90，两者可混用）。
- 跑完即删权重、留 manifest（rolling，参照上游 scheduler 的 cleanup 逻辑）。
- **产出**：`results/stage3_benchmark.json`：每臂每 seed 的 per-bench delta、六任务平均分、与 (a) 朴素随机 100% 的上述 20 组随机置换分布、(b) 噪声地板的对照。

### 阶段四：instruct 方向确认（成本：3 个 checkpoint）

- Qwen3-4B（instruct），每臂 1 个 seed（201），共 3 个 checkpoint，同协议跑 6 benchmark，baseline 用 `../ffn_benchmark_eval/results/raw/qwen3_4b__baseline_original_run1/`。
- 只回答一个问题：三臂的六任务平均分变化幅度排序在 instruct 上是否与 Base 一致。

## 7. Kernel 依赖性（必须写、不必测）

- 全部单层测量经由 `backend.py` 单点接口，切换 kernel 只改一处。
- `RESULT.md` 必须包含一节明确声明：**几何指标（尤其局部性/位移类）的结论依赖于本实验所用 GEMM kernel 的归约树结构（torch BF16 GEMM on RTX 4090 + vLLM batch-invariant kernel），未在其他 kernel/硬件上验证；跨 kernel 稳定性是留待后续的开放问题。**

## 8. 资源与共存约束（硬性）

1. **只用 GPU 0**。任何脚本启动前设置 `CUDA_VISIBLE_DEVICES=0`。
2. **与其他用户共存**：启动任何 >2GB 的任务前用 `nvidia-smi` 检查空闲显存；阶段二需 ~10GB、阶段三 vLLM 用 `gpu_memory_utilization=0.28`（~14GB）。空闲不足时等待重试：每 5 分钟检查一次，最多等 12 小时；OOM 失败自动重试（退避 45s，参照上游 run_worker 的重试实现），单任务最多 6 次重试；调度层再做最多 6 轮重扫。
3. 超过重试计划仍无法推进 → 按 §10.2 停止并报告，**不得**提高显存占用挤占别人。
4. conda 环境 `qwen3`；不安装/升级任何触碰 torch/vllm/transformers 的包。
5. 磁盘：阶段三峰值 1 个 checkpoint ≈ 7.6GB，rolling 删除；`results/` 预计 <200MB。

## 9. 验收标准（预注册，逐阶段核对）

### 阶段一通过 =（三条全部成立）

- **S1a**：非逐比特相等的测量点 ≥ 200 个，且预注册几何指标中**至少一个**在 log-log 回归达到 **R² ≥ 0.8** 或 **Spearman ρ ≥ 0.9**（三层与真实激活输入上分别成立，不允许只在合成输入成立）。
- **S1b**：F5（K=30% 恒定、只变 D）的漂移随 D **单调不减**（Spearman ρ(D, drift) ≥ 0.9）——证明位移距离的因果作用。
- **S1c**：F4-K30（相邻对换 30%）的单层漂移 < F3-K30（随机撒 30%）的 **1/3**（各取 5-seed median）。

### 阶段二通过 =（两条全部成立）

- **S2a**：24 个配置上，单层漂移（阶段一同构造实测值）与全模型 logits `rel_l2` 的 **Spearman ρ ≥ 0.8**。
- **S2b**：预测最小的 8 个配置中 ≥ 6 个落在全模型 `rel_l2` 的最低 8 名里。

### 阶段三通过 =（三条全部成立）

- **S3a**：六任务平均分变化幅度的 3-seed 均值满足 **min-cost < naive < worst** 排序。
- **S3b**：min-cost 臂的全部 3 seed：每个 benchmark 的 |Δ| ≤ 1pp，且 6 项 correctness agreement 均 ≥ naive 臂对应值。
- **S3c**：min-cost 臂的 logit 指标（对 9 个 checkpoint 顺带测阶段二协议的 32-prompt forward）优于 naive 臂（`rel_l2` 更小且 top-1 一致率更高）。

### 阶段四通过 =

- **S4**：instruct 上三臂的六任务平均分变化幅度排序 min-cost ≤ naive（允许 min-cost 与 naive 接近，但不得显著反序：min-cost > naive + 0.3pp 记为不通过）。

## 10. 未达标 / 失败时的行为（给执行 Agent 的固定指令）

### 10.1 科学性失败（验收不达标）

> 立即停止，不进入后续阶段。在实验根目录写 `FAILURE_REPORT.md`，包含：(1) 未通过的具体条款编号与实测数值 vs 阈值；(2) 已完成阶段的全部结果表；(3) 对失败原因的分析（数据支持的，不臆测）；(4) 若 S1a 失败且失败模式为"同几何结构、不同 seed 漂移方差大"，在报告中**建议**（不执行）引入数据感知指标作为后续方向；(5) 明确写"本轮实验到此停止，等待人工审阅"。**不要**调整阈值、增删 seed、换置换族后重跑。写完报告即结束任务。

### 10.2 基础设施失败

> GPU 等待超 12 小时 / 重试计划耗尽 / 磁盘不足：写 `STATUS_REPORT.md`，包含：(1) 阻塞的具体资源与时间线；(2) 已完成与未完成的任务清单（可断点续跑的状态）；(3) 恢复执行的准确命令。写完即停止。**不得**为继续执行而违反 §8。

### 10.3 结果异常但不构成失败

> 例如出现逐比特为零漂移的大片配置（预期内：F4/F5-D1 可能逐比特无漂移）：这是合法数据点，按 0 漂移记录，回归时归入"逐比特相等"类别单独报告，不视为异常。

## 11. 最终产物 `RESULT.md` 必须回答

1. 哪个几何指标最能预测单层 BF16 漂移？R²/ρ 是多少？定律的量化形式（漂移 ≈ f(指标) 的拟合式或查询表）。
2. 单层定律能否穿透到全模型 logits？（ρ、失配案例分析）
3. **"置换 30% 神经元的最小代价配方"**：具体构造 + 它在 logit 层与 benchmark 层相对 naive/worst 的实测差距 + 相对噪声地板的位置。
4. 定律对任意 K 的外推形式（从 F2/F3/F4 的 K 扫描拟合）。
5. Kernel 依赖性 caveat（§7 原文）。
6. 局限与后续（含：数据感知选择在什么证据下值得引入；对齐/merge 约束情形的推论——高贡献神经元的必要位移是否主导漂移，本轮不实验，只做基于定律的推演）。
7. 与预注册的全部偏离，逐条列出。

## 12. 目录结构

```
experiments/permutation_min_cost/
├── EXPERIMENT_PLAN.md        # 本文档（预注册，不得修改）
├── DECISIONS.md              # 执行中的实现级决策记录
├── scripts/
│   ├── permutation.py        # 已有：上游校验工具
│   ├── backend.py            # matmul 单点接口（kernel 切换预留）
│   ├── perm_families.py      # §4 八个族的构造 + 双射校验
│   ├── geometry.py           # §5 几何指标
│   ├── stage1_singlelayer.py
│   ├── stage2_fullmodel.py
│   ├── stage3_benchmark.py   # 生成 checkpoint + 调 ../ffn_benchmark_eval 的 worker
│   ├── stage4_instruct.py
│   └── analyze.py            # 回归、排序、验收核对（输出每条 S* 的 PASS/FAIL）
├── results/
│   ├── stage1_singlelayer.jsonl / stage1_regression.json
│   ├── stage2_fullmodel.jsonl
│   ├── stage3_benchmark.json
│   ├── stage4_instruct.json
│   ├── acceptance.json       # 每条验收条款的实测值与 PASS/FAIL
│   └── figures/
├── logs/
└── RESULT.md | FAILURE_REPORT.md | STATUS_REPORT.md
```

## 13. 参考路径速查

| 内容 | 路径 |
|---|---|
| 模型 | `/nvme0/if/models/Qwen3-4B-Base`、`/nvme0/if/models/Qwen3-4B`（只读） |
| 上游机制实验（probe 实现模式、prompts） | `/nvme0/if/permutation/experiments/ffn_permutation/` |
| benchmark 基建（worker/checkpoint/分析/冻结配置） | `/nvme0/if/permutation/experiments/ffn_benchmark_eval/scripts/`、`configs/frozen_config.json` |
| 冻结 baseline 原始结果（不重跑） | `../ffn_benchmark_eval/results/raw/qwen3_4b_base__baseline_original_run1/`、`.../qwen3_4b__baseline_original_run1/` |
| 朴素随机 100% 的 20 组随机置换参照分布 | `../ffn_benchmark_eval/results/stage2_distribution.json` |
| 噪声地板 | `../ffn_benchmark_eval/results/null_distribution.json` |
| 确定性配置说明 | `../ffn_benchmark_eval/RESULT.md` §1、memory `vllm-determinism-benchmark` |
