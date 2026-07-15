# ffn_benchmark_eval

FFN permutation 在 benchmark 层面的等价性与随机波动实验。上游 logits/activation 实验见
`../ffn_permutation/`。**结论见 [`RESULT.md`](RESULT.md)。**

## 回答的问题

1. 正确的全 FFN 联动 permutation（BF16 下）对 6 个 benchmark 的正确率影响有多大？
2. 影响与 permutation 的选取方式（局部/全局 scope、位移大小 magnitude）有什么关系？

## 结构

```
configs/       冻结的环境/模型/seed/样本/vLLM 配置 + 样本选择清单
bench/         评测基建与数据的硬拷贝（prompt builder / scorer / EvalPlus 数据）
scripts/
  common.py            复用 bench 的 prompt/scorer；路径与配置
  prepare_bench.py     修正数据路径 + 生成 500 样本确定性选择
  make_checkpoint.py   生成并校验 permuted / copy checkpoint（rolling）
  permutation.py       置换/校验工具（复用自 ffn_permutation）
  warm_evalplus.py     单进程预热 EvalPlus 缓存（避免并发竞争）
  run_worker.py        单 checkpoint 一次加载跑全部 6 benchmark（in-process vLLM）
  scheduler.py         GPU 调度 + rolling checkpoint + OOM 重试 + 分阶段
  analyze.py           配对分析：determinism / null 地板 / stage1 CI / stage2 分布 / 消融
  make_figures.py      seed 分布 / macro 直方 / 消融图
  probe_determinism.py 推理确定性探针（用于选定 batch-invariant 配置）
model_manifests/  原始模型 SHA-256
results/          raw/ 逐样本；*.json 聚合；figures/ 图
checkpoints/       仅保留轻量 permutation manifest，不含模型权重
logs/              运行时生成并忽略，不进入仓库
```

## 复现

```bash
conda activate qwen3
cd scripts
python prepare_bench.py            # 一次性样本选择
python scheduler.py --stage all --gpus 0   # 生成 checkpoint 并评测（可断点续跑）
python analyze.py --stage all      # 聚合分析
python make_figures.py             # 出图
```

关键配置（[`configs/frozen_config.json`](configs/frozen_config.json)）：`VLLM_BATCH_INVARIANT=1` +
`enforce_eager` + 关 prefix caching，每族固定单卡——这是获得逐 run 可复现 baseline 的前提，
否则 vLLM 长文本 greedy 生成本身会有可测的 run 间噪声。确定性证据见 [`RESULT.md`](RESULT.md) §1，冻结原因也记录在 [`configs/frozen_config.json`](configs/frozen_config.json) 的 `determinism_note`。

当前调度器仍包含原机器的模型、缓存与 GPU 路径假设；它们属于冻结复现环境的一部分。外部设备优先使用各探针的显式参数，新输出写入独立目录。完整 benchmark 的跨机器入口仍是 [`../../dev_list.md`](../../dev_list.md) 中的待办，未经检查不要直接启动全阶段 GPU 运行。
