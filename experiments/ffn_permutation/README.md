# Qwen3-4B FFN permutation 实验

对应预注册方案：[`ffn_permutation_experiment_plan.md`](../../docs/plans/ffn_permutation_experiment_plan.md)。
最终结论见 [`RESULT.md`](RESULT.md)；BF16 漂移来源的专题解释见
[`DRIFT_ORIGIN.md`](DRIFT_ORIGIN.md)。
Qwen3-4B-Base 的实验结果见 [`BASE_RESULT.md`](BASE_RESULT.md)。

## 复现

```bash
cd experiments/ffn_permutation
# 选一张空闲 GPU（约需 9 GiB 显存）
CUDA_VISIBLE_DEVICES=<free_gpu> bash run_all.sh
```

或分阶段：

```bash
conda run -n qwen3 python probe_synthetic.py            # A：小矩阵单测（≈20s，失败则退出码非 0）
conda run -n qwen3 python probe_single_mlp.py --resume  # B：单 MLP 隔离（≈5min）
conda run -n qwen3 python probe_full_model.py --resume  # C：全模型（≈30-60min；--smoke 可快速冒烟）
conda run -n qwen3 python summarize.py                  # 汇总 JSON
```

## 文件

| 文件 | 内容 |
|---|---|
| `permutation.py` | perm 构造 / in-place apply / inverse restore / sha256 / 指标 |
| `probe_synthetic.py` | A 层：d=7,m=11 BF16 + fp64 方向性与判据单测 |
| `probe_single_mlp.py` | B 层：layer 0/17/35 真实 MLP，10 组对照 × 5 seed × 5 输入 |
| `probe_full_model.py` | C 层：32 prompt 全模型传播 + 负对照 + greedy generation |
| `prompts.json` | 固定 32 条中英文 prompt |
| `results/tokenized.json` | 一次性 tokenize 结果，所有 case 复用 |
| `results/*.jsonl` | 逐 case 原始记录（含 restore 校验） |
| `results/*manifest*.json` | case 状态与环境信息 |
| `summarize.py` | 聚合出 RESULT.md 所用的统计 |

## 约定

- permutation 定义：`z_perm = z[..., perm]`；权重侧
  `gate/up: w[perm, :]`，`down: w[:, perm]`；inverse 用 `argsort(perm)`。
- 所有权重修改均在内存中进行，每个 case 后 inverse 还原并与 CPU master
  copy 逐字节比对 + SHA-256 校验；不生成任何 permuted checkpoint。
- 模型目录只读；本实验不依赖其他实验目录中的实现。
