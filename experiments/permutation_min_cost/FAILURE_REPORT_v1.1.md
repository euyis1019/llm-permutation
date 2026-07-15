# 修订案 v1.1 科学性失败报告

> 日期：2026-07-12  
> 有效预注册：`AMENDMENT_v1.1.md`  
> 终止点：阶段 1b，硬判据 S1b-1 失败；阶段 2b/3b 未启动

## 1. 终止结论

阶段 1b 已完整产生 **1,248/1,248** 条互异测量（52 实例 × 3 层 × 2 prompt × 2 backend × 2 形状）。硬判据 **S1b-1** 要求全部预测“免费”的测量 100% 逐比特相同；实测只有 **431/504 = 85.5159%** 逐比特相同，**73 条失败**，因此未达到 100% 阈值。

依 `AMENDMENT_v1.1.md` §B/§E，首个硬判据失败后立即停止。本轮没有启动阶段 2b 的 vLLM 引擎端到端测量，也没有创建或评测阶段 3b checkpoint；没有调整阈值、seed、族或标签规则，也没有重跑正式测量。

## 2. 阶段 1b 全部验收结果

| 条款 | 阈值 | 实测 | 判定 |
|---|---:|---:|---|
| **S1b-1（硬）** | 免费预测 100% bitwise equal | 431/504 = **85.5159%**；73 条失败 | **FAIL / 停机** |
| S1b-2 | 饱和预测 ≥95% 同时非 bitwise 且归入 ceil | 373/624 = **59.7756%**；单独非 bitwise 率 89.7436% | FAIL |
| S1b-3 | 每个 backend×形状×层的实测 ceil 档 p95/p5 ≤3 | torch/full 三层通过；torch/decode 三层无 ceil 记录；vLLM 六组比值 4.153–7.559 | FAIL |
| S1b-4 | 三档分类准确率 ≥85% | 874/1248 = **70.0321%** | FAIL |
| S1b-5 | 记录项，无阈值 | torch ceil 中位数 2.9434524e-3；vLLM 7.0132750e-5；torch/vLLM = **41.9697** | 记录 |

实测档位的混淆矩阵（行是预注册预测，列是实测 zero/sub/ceil）：

| 预测 \ 实测 | zero | sub | ceil | 合计 |
|---|---:|---:|---:|---:|
| zero | 431 | 73 | 0 | 504 |
| sub | 17 | 70 | 33 | 120 |
| ceil | 64 | 187 | 373 | 624 |
| 合计 | 512 | 330 | 406 | 1248 |

实测归档严格使用预注册规则：zero = `bitwise_equal`；torch 的 sub 上界为 `3e-4`；vLLM 的“相应 backend 中位天花板”实现级定义为该 backend 全部预注册 predicted-ceil 记录的 `rel_l2` 中位数（5.7395589e-5），故 sub 上界为其 1/3，即 1.9131863e-5。此实现选择已在 `DECISIONS.md` 记录；它不参与 S1b-1 的逐比特判定。

### S1b-1 完整分解

| 预测免费族 | backend | 形状 | bitwise / n | 失败数 | 最大 rel_l2 | 最大不同元素数 / 2560 |
|---|---|---|---:|---:|---:|---:|
| F9 | torch_bf16 | full | 90/90 | 0 | 0 | 0 |
| F9 | torch_bf16 | decode1 | 37/90 | **53** | 5.9845264e-5 | 3 |
| F9 | vllm_bi | full | 90/90 | 0 | 0 | 0 |
| F9 | vllm_bi | decode1 | 90/90 | 0 | 0 | 0 |
| F11-o0 | torch_bf16 | full | 30/30 | 0 | 0 | 0 |
| F11-o0 | torch_bf16 | decode1 | 10/30 | **20** | 5.9845264e-5 | 3 |
| F11-o0 | vllm_bi | full | 30/30 | 0 | 0 | 0 |
| F11-o0 | vllm_bi | decode1 | 30/30 | 0 | 0 | 0 |
| F8 identity | torch_bf16 | full | 6/6 | 0 | 0 | 0 |
| F8 identity | torch_bf16 | decode1 | 6/6 | 0 | 0 | 0 |
| F8 identity | vllm_bi | full | 6/6 | 0 | 0 | 0 |
| F8 identity | vllm_bi | decode1 | 6/6 | 0 | 0 | 0 |

73 条失败覆盖全部新 seed 301–305；按层为 L0/L17/L35 = 33/20/20，按 prompt 24/0 = 46/27。每条失败只有 1–3/2560 个输出元素不同，`rel_l2` 范围为 1.1470443e-9 至 5.9845264e-5，但 S1b-1 是逐比特硬判据，因此幅度再小也构成失败。

首条失败位于原始文件第 2 行：`F9 K=5%, seed=301, L0, prompt=24, torch_bf16, T=1`，`n_diff=2/2560`，`rel_l2=1.6166857e-7`。最大漂移之一位于第 282 行：`F9 K=100%, seed=302, L35, prompt=24, torch_bf16, T=1`，`n_diff=3/2560`，`rel_l2=5.9845264e-5`。

### S1b-3 的 12 个分组

| backend | 形状 | 层 | ceil n | p5 | p95 | p95/p5 | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| torch_bf16 | full | 0 | 42 | 2.7454e-3 | 3.0431e-3 | 1.108 | PASS |
| torch_bf16 | full | 17 | 42 | 3.1046e-3 | 3.2356e-3 | 1.042 | PASS |
| torch_bf16 | full | 35 | 42 | 1.3405e-3 | 2.9245e-3 | 2.182 | PASS |
| torch_bf16 | decode1 | 0 | 0 | — | — | — | FAIL（无 ceil 记录） |
| torch_bf16 | decode1 | 17 | 0 | — | — | — | FAIL（无 ceil 记录） |
| torch_bf16 | decode1 | 35 | 0 | — | — | — | FAIL（无 ceil 记录） |
| vllm_bi | full | 0 | 59 | 2.8304e-5 | 1.8170e-4 | 6.419 | FAIL |
| vllm_bi | full | 17 | 62 | 3.5328e-5 | 1.9420e-4 | 5.497 | FAIL |
| vllm_bi | full | 35 | 61 | 2.2255e-5 | 1.6822e-4 | 7.559 | FAIL |
| vllm_bi | decode1 | 0 | 40 | 3.7086e-5 | 1.8265e-4 | 4.925 | FAIL |
| vllm_bi | decode1 | 17 | 34 | 3.7258e-5 | 2.1285e-4 | 5.713 | FAIL |
| vllm_bi | decode1 | 35 | 24 | 2.8633e-5 | 1.1890e-4 | 4.153 | FAIL |

## 3. 数据支持的失败模式分析

失败并非遍布所有条件，而是**完全局限于 `torch_bf16 × T=1`**：

- torch/full：126/126 个免费预测逐比特相同；
- vLLM/full：126/126；vLLM/decode1：126/126；
- torch/decode1：仅 53/126，另有 73 条失败；
- identity 在全部 backend、形状、层、prompt 上 24/24 逐比特相同，排除了 baseline 自身不确定或比较管线普遍失效；
- 两个非恒等免费族 F9 与 F11-o0 都在 torch/decode1 失败，且覆盖全部五个新 seed，排除了单一族或单一 seed 的偶发现象。

因此数据只支持以下结论：**“保持对齐 8-块成员关系 ⇒ 逐比特免费”在本轮的 torch BF16 full/prefill 与 vLLM batch-invariant 两种形状上成立，但不覆盖本轮实际调用的 torch BF16 `F.linear` T=1 路径。** 这种严格集中于 backend×形状的分裂与 kernel 归约微结构依赖相容；仅凭本轮记录不能进一步定位 torch 内部选择了哪个具体算子或归约布局，故不作超出数据的根因断言。

非硬判据也显示相同的形状错配：torch/decode1 的饱和预测全部低于固定 3e-4 ceil 门槛，三层都没有实测 ceil 记录；这同时压低了 S1b-2 和总分类准确率。vLLM 的实测 ceil 分布也比预注册的三倍范围更宽。以上均是正式数据结果，没有用于修改硬停机决定。

## 4. 复现命令与原始记录

正式测量实际执行命令（输出已存在，脚本会拒绝静默覆盖）：

```bash
cd /nvme0/if/permutation
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  /nvme0/if/anaconda3/envs/qwen3/bin/python \
  experiments/permutation_min_cost/scripts/stage1b_singlelayer.py
```

在隔离目录完整复现的命令（**本轮停机后未执行**，仅供人工复核）：

```bash
cd /nvme0/if/permutation
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  /nvme0/if/anaconda3/envs/qwen3/bin/python \
  experiments/permutation_min_cost/scripts/stage1b_singlelayer.py \
  --model-path /nvme0/if/models/Qwen3-4B-Base \
  --results-dir experiments/permutation_min_cost/reproduction_v11_stage1b
```

从原始记录重建验收（只分析、不重新测量）：

```bash
cd /nvme0/if/permutation
/nvme0/if/anaconda3/envs/qwen3/bin/python \
  experiments/permutation_min_cost/scripts/analyze_v11.py
```

| 产物 | 作用 | SHA-256 |
|---|---|---|
| `results/stage1b_singlelayer.jsonl` | 1248 条不可变原始测量 | `d524b3cdece6b7996781bea8bd071ce9edb8d6f5696c98019c64ec4e5bef3197` |
| `results/stage1b_classified.jsonl` | 逐条预测/实测档位与阈值 | `719544228be80260fc909f3bf4ed9e86ff06042c91913ee434665ed4fbd53c1c` |
| `results/stage1b_free_failures.jsonl` | 73 条 S1b-1 失败完整记录（含原始行号） | `bf29ca64485386b502e71a3674e04328aa74d1e55a568ee12cf446e672ad0da4` |
| `results/stage1b_manifest.json` | 完整性、prompt/token、环境清单 | `ccf73a6561c73ffd45620aaa8070a0702874a665838c32429e6acfa75fb8ea60` |
| `results/acceptance_v11.json` | S1b-1..S1b-5 机器可读验收 | `5f3880cce92a615e86d86f800e1520ca9544ce3fc84ade076d0f040c1f7c847c` |

环境：Qwen3-4B-Base，RTX 4090，GPU 0，torch 2.11.0+cu130，CUDA 13.0，transformers 5.13.0，vLLM 0.24.0；正式测量耗时 33.1 秒。

## 5. 后续阶段状态

| 阶段 | 状态 | 原因 |
|---|---|---|
| 1b | 完成；硬判据失败 | S1b-1 = 431/504，而要求 504/504 |
| 2b | **未执行** | 被 §E 硬停机禁止 |
| 3b | **未执行** | 被 §E 硬停机禁止 |

**本轮实验到此停止，等待人工审阅。**
