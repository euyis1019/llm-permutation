# 实验三：activation / 运算量级画像

Qwen3-4B-Base 的实验结果见 [`BASE_RESULT.md`](BASE_RESULT.md)。

测量 Qwen3-4B（BF16）各层各流的真实数值量级，并与 BF16 格点刻度对照，
为 permutation 漂移给出定量闭环。报告见 [`RESULT.md`](RESULT.md)。

- 冻结运行输入：原 `/nvme0/if/llm-brewing/bench/datasets/benchmark/normalized` 的 9 个基准；仓库内保留的副本可通过 `--bench-dir ../ffn_benchmark_eval/bench/datasets/benchmark/normalized` 显式传入
  各随机采 16 条（seed 42），共 144 prompt / 16132 token；
- 测量流：embed、残差流、attn_out、mlp_in、gate/up 输出、h（down 输入）、
  mlp_out、final_norm、logits，每流 RMS / abs_mean / p50 / p99 / max；
- down-GEMM 深潜（L0/17/35）：逐项对消比、最大单项占比、bf16 vs fp32 vs fp64
  的三层误差分解（`results/down_gemm_decomposition.json`）；
- 模型只读，不修改任何权重。

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> conda run -n qwen3 python probe_magnitudes.py \
  --bench-dir ../ffn_benchmark_eval/bench/datasets/benchmark/normalized \
  --results-dir <new_results_dir>
```
