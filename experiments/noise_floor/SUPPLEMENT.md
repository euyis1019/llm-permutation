# noise_floor 补充实验(post-hoc,reviewer 发起)

> 日期:2026-07-13。预注册的 Part 0/1/2/6 已完成且三条硬判据全部通过。
> 本文件记录的三个补充臂是在看过主结果之后追加的,属于 post-hoc 探索,
> 结论权重低于预注册部分,写报告时必须标明。
> 复用冻结的 run_logits_unit.py / run_benchmark_unit.py,资源纪律同 EXPERIMENT_PLAN.md §1。
> 执行脚本:scripts/run_supp_units.py;加噪:scripts/supp_make_noise_checkpoint.py。

## 动机

主结果 Part 6b 显示 σ=1e-6 的全模型高斯扰动 logits 漂移中位数已达 F7 全 36 层
跨块置换参考线的 1.0065 倍,σ* 只能记为"不高于最低网格点"。这留下三个洞:

1. 地板的左边缘没有定位:再往下多小的 σ 才会脱离地板,是否存在全部被 bf16
   量化吞掉、逐比特一致的档位?
2. 口径混淆:Part 6b 的噪声加在全模型所有参数上(含 embedding、lm_head、norm),
   而置换只动 36 层 FFN 的三个投影矩阵。地板一致有可能是 embedding 扰动贡献的。
3. 行为层缺口:6c 因 σ* 出界被条件跳过,但"RandOpt 最小档 σ 在 benchmark 上的
   影响是否超出置换零假设分布"恰恰是本工作对外最重要的一句话,值得补测。

## 三个臂

| 臂 | 作用域 | σ | seed | 测量 | 单元数 |
|---|---|---|---|---|---|
| S1 左边缘 | 全模型 | 1e-8, 1e-7, 3e-7 | 3000+10·idx+rep,rep∈{0,1,2} | 32 prompt logits | 9 |
| S2 口径对齐 | 仅 36 层 FFN gate/up/down | 1e-6, 1e-5, 1e-4 | 4000+10·idx+rep | 32 prompt logits | 9 |
| S3 行为臂 | 全模型 | 1e-4, 1e-3 | 5000+100·idx+rep,rep∈{0..4} | GSM8K-500 correctness | 10 |

其余口径与 Part 6b 完全一致:同 32 条冻结 prompt、同 vLLM 冻结配置、同
"torch.randn 原 dtype 原位相加"的加噪方式、一次只存在一个临时 checkpoint。

## 跑之前写下的预期

- S1:预计任何档都不会出现逐比特一致(bf16 的 ulp 随权重大小缩放,接近零的
  权重总会被改到)。真正的问题是漂移是否随被改参数比例下降:如果 σ=1e-8 仍在
  地板上,结论升级为"哪怕只改动极小比例的权重、每个只改一个 ulp,36 层之后
  也会到达同一块地板";如果下降,就定位了地板的左边缘。
- S2:预计 FFN-only 的 σ=1e-6 同样落在地板附近。如果显著更低,说明全模型曲线
  高估了与置换同口径的扰动,主报告的谱系图必须改用 S2 口径。
- S3:预计 σ=1e-4 的 5 个 seed 的 GSM8K 准确率落在 20 组随机置换（由 20 个不同随机种子各生成一套置换）构成的零假设分布的
  范围内(即与"功能完全不变的模型"不可区分);σ=1e-3 方向不定,可能开始出界。

## 结果文件

- results/supp_units/<tag>/summary.json + logits/(S1、S2)
- results/supp_behavior/<tag>/gsm8k.raw.json(S3)
- results/supp_weight_stats/<tag>.json(每个 checkpoint 的权重级统计)
- reviewer_analysis/supp_summary.json(analyze_supp.py 产出)
