# 实验二：FFN 前后半块交换（half-swap）

Qwen3-4B-Base 的实验结果见 [`BASE_RESULT.md`](BASE_RESULT.md)。

对 Qwen3-4B 全部 36 层 FFN 的 intermediate 轴（m=9728）做**前一半 ↔ 后一半**
的块交换（`perm = [4864..9727, 0..4863]`，自逆置换），按实验一验证过的联动方式
（gate/up 行 + down 列，同一 perm）施加。

## 观测协议

- 输入：复用实验一的 32 条固定 prompt（`../ffn_permutation/results/tokenized.json`，
  逐 bit 相同的 input_ids）；
- forward：每条件重复 2 次（确定性验证），对比全 token logits 与 last-token logits；
- 生成：温度 0（`do_sample=False` 贪心，64 tokens），**每条件独立重复 8 次**——
  先验证条件内 8 次逐 token 一致，再对比 baseline vs half-swap；
- 权重 in-place 置换，结束后 inverse 还原并与 CPU master copy 逐字节比对 + 全 MLP SHA-256。

## 运行

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> conda run -n qwen3 python probe_halfswap.py
```

结果：`results/halfswap.jsonl`（逐 prompt）、`results/manifest.json`；结论见 `RESULT.md`。
