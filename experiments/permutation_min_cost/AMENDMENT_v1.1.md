# 修订案 v1.1：三档门控-饱和定律的预注册确认（替代原阶段二至四）

> 生效日期：2026-07-12
> 地位：本文件是**唯一有效**的后续预注册。原 `EXPERIMENT_PLAN.md` 的 §0（总指令）、§2（环境）、§8（GPU 0 资源约束）、§10（停机 prompt）、§11（DECISIONS.md 记录规则）**全部继承**；原阶段二/三/四作废。
> 背景：阶段一按原判据 FAIL（正确停机）。复审（`review/REVIEW_stage1.md`）确立了修正假说：单层 BF16 漂移不是连续定律，而是三档门控-饱和结构，免费档 = 保持对齐 8-块成员关系。本修订案对该假说做**全新数据**上的预注册确认，并把它推到端到端与 benchmark 层。复审的事后分析不作为结论，本文件的验收结果才是。

## A. 冻结的新增置换族（seed 固定 301–305）

M=9728。所有族用 `scripts/perm_families.py` 同样的确定性构造风格实现，新增：

| 族 | 构造 | 参数 |
|---|---|---|
| F9_inblock_shuffle | 随机选 ⌈K·M/8⌉ 个**对齐 8-块**，各块内部随机重排（保证块内非恒等） | K ∈ {5%, 30%, 100%} |
| F10_odd_pairs | 在奇对齐候选对 {(2k+1, 2k+2)} 中随机选 ⌈K·M/2⌉ 对交换（distance-1 但跨对齐块） | K ∈ {30%, 100%}（K=100% 无随机性，1 个实例） |
| F11_window_offset | 宽 8 窗内乱序，窗起点整体偏移 o | o ∈ {0, 4} |
| F12_win16_aligned | 对齐 16-窗内乱序 | — |

对照族沿用原实现：F3_scattered_global（K ∈ {5%, 30%}）、F7_global_random、F8_identity。

实例数：F9 3×5 + F10 (5+1) + F11 2×5 + F12 5 + F3 2×5 + F7 5 + F8 1 = **52**。

## B. 阶段 1b：单层三档定律确认（全新数据）

- 输入：真实激活，**两个 prompt**（id=24 与最小的 id≠24），L0/L17/L35。合成输入不再使用。
- backend：`torch_bf16` 与 `vllm_bi`（`vllm.model_executor.layers.batch_invariant.linear_batch_invariant`），运行时语义 `y = F.linear(x[:,p], W[:,p])`，x 形状 [T, M]，T ∈ {124 截断实 token 数, 1}。
- 测量：52 实例 × 3 层 × 2 prompt × 2 backend × 2 形状 ≈ **1248 条**（GPU 0，分钟级）。
- 每条记录三档预测标签（规则预注册如下）与实测 rel_l2 / bitwise_equal：
  - 预测"免费"：∀i: ⌊i/8⌋ = ⌊π(i)/8⌋（F9、F11-o0、F8_identity）
  - 预测"亚 ulp"：非免费且 ∀i: ⌊i/16⌋ = ⌊π(i)/16⌋（F12）
  - 预测"饱和"：其余（F10、F11-o4、F3、F7）
  - 实测归档：zero = bitwise_equal；sub = 0 < rel_l2 < 3×10⁻⁴（torch）/ 相应 backend 中位天花板的 1/3；ceil = 其余

### 验收（阶段 1b）

- **S1b-1（免费档，硬判据）**：所有预测"免费"的测量 **100% bitwise_equal**，两 backend、两形状、三层、两 prompt 无一例外。任何一条失败即证伪免费类 → 停机走 §10.1。
- **S1b-2（非免费档）**：预测"饱和"的测量 ≥95% 非 bitwise_equal 且落入 ceil 档。
- **S1b-3（天花板普适性）**：每个 backend×形状×层 内，ceil 档 rel_l2 的 p95/p5 ≤ 3。
- **S1b-4（三档分类准确率）**：全部测量上规则分类准确率 ≥ 85%（预期误差集中在 sub/ceil 边界，允许）。
- **S1b-5（kernel 天花板差异记录项，不设阈值）**：报告两 backend 天花板中位数与比值。

## C. 阶段 2b：端到端逐比特验证（vLLM 引擎）

配置：Qwen3-4B-Base，GPU 0，`VLLM_BATCH_INVARIANT=1` + enforce_eager + 无 prefix caching，`gpu_memory_utilization=0.28`，原 §8 资源规则。32 prompt（上游冻结集）取 prompt-last-token logits。

4 个 checkpoint（in-memory 置换即可，不落盘权重）：identity；F9-K100-all36（每层独立 seed=401+layer）；F10-K100-all36；F7-all36（seed=402+layer）。

### 验收（阶段 2b）

- **S2b-1（硬判据）**：F9-K100-all36 与 identity 的 logits **逐比特相同**（max|Δ|=0，全 32 prompt）。失败 ⇒ 引擎内还存在破坏免费类的算子 → 停机报告，定位到算子不在本轮范围。
- **S2b-2**：F10 与 F7 的 logits 均非逐比特相同，且两者 rel_l2 同数量级（比值 ∈ [1/5, 5]，中位数意义）。

## D. 阶段 3b：benchmark 终验（3 checkpoint，只终验不搜索）

沿用 `ffn_benchmark_eval` 全套冻结基建（6 benchmark、500 样本、determinism 配置、既有 base baseline 直接复用，null 地板 = 0）。3 个落盘 checkpoint（rolling，评完即删权重）：

1. **F9-K100-all36**（免费臂）
2. **F10-K100-all36**（锋利预测臂：distance-1 但跨块 ⇒ 预测效应与随机置换同量级）
3. **F3-K30-all36, seed 301**（对照臂：原始问题里的"朴素散置 30%"）

### 验收（阶段 3b）

- **S3b-1（硬判据）**：免费臂 6 个 benchmark Δ 恰为 0、文本一致率 100%（base null 地板为 0，任何一条样本级差异即失败 → 停机报告）。
- **S3b-2**：F10 臂在 ≥2 个生成类 benchmark 上行为不一致率 > 0，且六任务平均分变化幅度与 F3 臂之比 ∈ [1/3, 3]。
- **S3b-3（记录项）**：F3-K30 臂与 `ffn_benchmark_eval` 的 20 组随机置换（由 20 个不同随机种子各生成一套置换）所形成分布的位置关系（预期落入其范围内）。

## E. 停机与产出

- 三个硬判据（S1b-1、S2b-1、S3b-1）任一失败：立即停止后续阶段，写 `FAILURE_REPORT_v1.1.md`（含失败测量的完整复现命令与原始记录路径），不调整规则不重跑。
- 全部通过：写 `RESULT.md`，内容必须包含：三档定律表（两 backend）、免费类的形式刻画、端到端逐比特结论、benchmark 三臂表、以及 §7 继承的 kernel 依赖 caveat（明确写出：免费类的边界是 kernel 归约微结构的属性，本结论覆盖 torch bf16 GEMM 与 vLLM 0.24 batch-invariant 两个 backend，RTX 4090；换 kernel/硬件需用 `backend.py` 重跑阶段 1b，跑之前不得外推）。
- 逐阶段照常更新 `results/acceptance_v11.json` 与 `DECISIONS.md`。

## F. 预算

阶段 1b 分钟级；阶段 2b 单次引擎加载 ~10 分钟级;阶段 3b 3 checkpoint × 6 benchmark，参照上一实验单 checkpoint ~1–1.5 h（GPU 0 共享、0.28 util），合计 ≤ 6 h GPU 时间。全程 GPU 0 且遵守原 §8 的显存检查/等待/退避规则。
