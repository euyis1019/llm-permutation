# 一个最小的参数扰动框架

这不是准备投入训练的 library，而是一份可执行语法检查的“概念代码”。它来自对当前目录中 RandOpt、CoRP、ES at Scale、EGGROLL、QES/QZO、DiZO、FZOO、SubZero、LOZO 和 ZO Fine-tuner 核心路径的交叉阅读。

最重要的判断是：**存在稳定抽象，但公共基类不应是 `Optimizer`，也不应假设中间产物是 dense gradient。** 更稳的共同循环是：

```text
                  proposal/search state（可在拒绝后继续变化）
                                  │
                                  ▼
anchor θ ── ask ──> Candidate(anchor, symbolic Direction)
                         │
                         ▼
              transactional materialize
                 θ + Δᵢ → forward → utilityᵢ
                         │               │
                         └── exact reset ┘
                                  │
                                  ▼
                      reduce / consolidate
                                  │
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                  ▼
        WeightDeltaPlan     CommitteePlan       DistillPlan
          ZO/ES/CoRP          RandOpt        iterative RandOpt
               └──────────────────┼──────────────────┘
                                  ▼
                         optional held-out gate
                                  ▼
                                commit
```

一句话版本是：

```text
proposal → black-box score → score-to-plan → commit
```

真正不可被抹平、必须留给策略或 backend 的，是三件事：

1. proposal 怎样作用到模型：原位浮点加法、functional low-rank operator、量化 scale，还是 unpack/mutate/repack 的整数 code；
2. score 怎样变成系数：中央差分、one-sided baseline、population z-score、rank/elite、group normalization；
3. commit 落到什么对象：连续权重、带 residual/rounding 的整数格点、输出 committee，还是通过 SFT/KD 得到新 checkpoint。

## 目录

```text
minimal_perturbation_framework/
├── README.md
├── examples.py                         # 八种方法如何由少量组件拼出来
├── perturbation_framework/
│   ├── core.py                         # NoiseRef / Direction / Candidate / Plan
│   ├── policies.py                     # ask：两点、population、FZOO、低秩/量化 family
│   ├── reducers.py                     # score → weight delta / committee / teachers
│   └── runtime.py                      # evaluate transaction、gate、commit、StepEngine
└── tests/test_mental_model.py          # 小型结构测试，不涉及 tensor
```

阅读顺序建议：先看 [`core.py`](./perturbation_framework/core.py)，再看 [`examples.py`](./examples.py)，最后看 reducer 公式。

## 六个核心对象

### 1. `NoiseRef`：可重放的单位方向

`seed` 本身不够。一个完整 replay key 还必须说明：

- direction family：Gaussian、Rademacher、low-rank、quantized rounding；
- target space：float weight、adapter、quant scale、integer code、functional operator；
- parameter scope：哪些参数参与；
- RNG scheme/version：全局连续 stream、逐 tensor 重置，还是按参数名 hash；
- family options：rank、bit width、rounding stream 等。

`sigma` 不在 `NoiseRef` 内，而是 `DirectionTerm.amplitude`。这样同一 seed 在不同 sigma 下仍然是同一个方向，线性组合也不会误把它们当成不同 basis。

### 2. `Direction`：lazy linear combination

```python
Direction = Σ amplitude_j × NoiseRef_j
```

它是一个小型 IR，不是 tensor。RandOpt 一个候选通常只有一个 term；ES/ZO 更新是多个 replay key 的加权和；CoRP merge 可能是几百个 term。backend 在逐参数 streaming 时才重建噪声。

对于 `INTEGER_CODE`，这个线性和表示的是**投影/舍入之前的更新命令**，并不声称整数格点上的 materialization 是线性的。QES 的 rounding、boundary mask、anti-windup 和 residual 必须由该 target space 的 backend 持有。

### 3. `Candidate` 与 `Trial`

`Candidate` 是 `anchor + Direction`，并把 `pair_id`、`role=plus/minus/baseline`、group 等 population layout 作为一等信息保存。EGGROLL/QES 的 antithetic 关系不能靠“seed 是正还是负”的偶然编码推断。

`Trial` 是 candidate 的观测，统一使用 higher-is-better `utility`。若任务返回 CE loss，objective adapter 负责转成 `-loss`。它还可保存 per-example utility 和 output reference：

- per-example score 让 CoRP 计算 “fixes − regressions” 或 held-out gate；
- output reference 让 RandOpt committee 在部署时重放或聚合回答。

### 4. `SearchState`

它显式分开：

- `anchor_id`：当前模型中心；
- `search_state`：当前 sigma、低秩 basis version、learned scales、subspace statistics 等。

这不是形式主义。CoRP 的 proposal 即使被 gate 拒绝，搜索分布仍可能根据本轮 trial 调整；anchor 保持不动。把两者塞进一个 optimizer state 会掩盖这个行为。

### 5. `DeploymentPlan`

统一循环的输出不是统一 gradient，而是一个 sum type：

- `WeightDeltaPlan`：ZO、ES、CoRP；
- `CommitteePlan`：原始 RandOpt top-k majority vote；
- `DistillPlan`：iterative RandOpt 的 SFT/KD。它明确告诉读者，整个 pipeline 不再是 gradient-free；
- `RejectPlan`：held-out validation 不通过。

### 6. `CandidateBackend`

`CandidateBackend.candidate(anchor, direction)` 返回 context manager。唯一强制不变量是：退出 context 后恢复**精确 anchor**。

```python
with backend.candidate(anchor_id, direction) as model_view:
    utility = objective.score(model_view)
# 即使 score 抛异常，这里也必须已回到 anchor
```

Ray/vLLM 可在这里做 snapshot/materialize/reset；JAX/EGGROLL 可以返回 functional operator view；QES 可以携带 unpack 后的 boundary token。执行并行度不是算法核心，所以 `SequentialEvaluator` 可以被 Ray/JAX evaluator 整体替换，而其余对象不变。

## 几个方法只是换了哪一块？

| 方法 | `ask` / direction | score reducer | deployment |
|---|---|---|---|
| MeZO / QZO | 同一 key 的 `±σ` pair | central difference | weight delta；QZO target 是 quant scale |
| FZOO | N 个 Rademacher 单边方向 + baseline | `(uᵢ-u₀)/(N·std)` | weight delta |
| ES at Scale | one-sided Gaussian population | global z-score | replayed weighted delta |
| EGGROLL | antithetic low-rank functional directions | global/group shaping | Optax/dense-equivalent delta |
| QES | bounded integer-code proposals | z-score 或 rank/elite；可 mirror | residual + stochastic rounding commit |
| RandOpt | 多 sigma 的 one-sided population | top-k | output committee，不移动 anchor |
| CoRP | RandOpt population，后续低秩+isotropic local proposals | reward + alignment − dispersion | merged delta + validation gate |
| iterative RandOpt | 小 population | top-k teachers | SFT/KD 新 anchor |
| SubZero / LOZO | stateful low-rank direction policy | 通常 central difference | streaming SGD-like delta |
| DiZO | 外层仍是 MeZO Gaussian | 外层 central difference | 周期性 post-step radius controller |

[`examples.py`](./examples.py) 把前八项写成实际对象组合。例如 RandOpt 与 ES 的 `Population` 可以完全一样，只有 consolidator 不同：

```python
# ES：候选被压成一个权重更新
Population(...), PopulationES(...)

# RandOpt：候选本身成为部署 artifact
Population(...), TopKCommittee(top_k=50)
```

而 EGGROLL 和普通 ES 的控制面也可以一样，只需换 lazy direction family：

```python
Population(..., family=NoiseFamily("isotropic_gaussian"))
Population(..., family=low_rank_family(rank=1))
```

这正是“合理抽象”的证据：policy/reducer 不需要知道低秩方向是通过 `W+ABᵀ` materialize，还是在 matmul 中直接计算 `xWᵀ+xBAᵀ`。

## 代码阅读依据

### RandOpt / CoRP

- RandOpt 生成 `(seed, sigma)`、依次 perturb/generate/restore：[randopt.py](https://github.com/sunrainyg/RandOpt/blob/536df0a308f3/randopt.py#L140)。
- top-k 候选在测试时逐个重放并多数投票：[randopt.py](https://github.com/sunrainyg/RandOpt/blob/536df0a308f3/randopt.py#L203)。
- worker 以 seed 重建噪声并原位加减：[worker_extn.py](https://github.com/sunrainyg/RandOpt/blob/536df0a308f3/utils/worker_extn.py#L69)。
- CoRP 已经出现 `DirectionComponent` 和线性组合 IR：[corp_ops.py](https://github.com/oooranz/CoRP/blob/4cba6bf9102d/core/corp_ops.py#L9)。
- CoRP 的 elite、两遍加权、alignment/dispersion 与 PCA：[collapse_ops.py](https://github.com/oooranz/CoRP/blob/4cba6bf9102d/core/collapse_ops.py#L137)。
- principal subspace 与 isotropic residual 的下一轮采样：[recenter_ops.py](https://github.com/oooranz/CoRP/blob/4cba6bf9102d/core/recenter_ops.py#L8)。

### ZO variants

- FZOO 的 Rademacher direction、N 次单边 probe、一次 baseline 与逐 seed 更新：[trainer.py](https://github.com/DKmiyan/FZOO/blob/d4f8a8b16eb6/trainer.py#L722)。
- SubZero 的持久化 `U,V` 子空间与每步小矩阵 direction：[trainer.py](https://github.com/zimingyy/SubZero/blob/cf6effdd0aef/large_models/trainer.py#L744)。
- LOZO 固定/周期更新右因子 `V`、每步采 `U`：[LOZOtrainer.py](https://github.com/optsuite/LOZO/blob/5d1ade5bf418/large_models/LOZOtrainer.py#L699)。
- DiZO 外层 MeZO probe：[trainer.py](https://github.com/Skilteee/DiZO/blob/b05e73d2016f/large_models/trainer.py#L1335)；其独特部分是 post-step 的 layer radius controller，而非另一种外层噪声。
- ZO Fine-tuner 按参数块 history 预测不同 scale：[trainer_zo_fine_tuner.py](https://github.com/ASTRAL-Group/ZO_Fine_tuner/blob/ef9abff5f74d/trainer_zo_fine_tuner.py#L618)。

### ES / low-rank / quantization

- ES at Scale 的 population seeds、reward z-score 与 seed-replayed FP32 accumulation：[es_trainer.py](https://github.com/VsonicV/es-fine-tuning-paper/blob/574a9d134da1/es_at_scale/trainer/es_trainer.py#L423)。
- EGGROLL 已把 noiser hook 从模型算子中抽出；低秩 matmul 不需构造完整 `W+ΔW`：[eggroll.py](https://github.com/ESHyperscale/HyperscaleES/blob/b77f7d6f9123/src/hyperscalees/noiser/eggroll.py#L77)。
- QES mirror pair、z-score/rank shaping 与 update dispatch：[int4_perturb.py](https://github.com/dibbla/Quantized-Evolution-Strategies/blob/fefc7358decb/int4_perturb.py#L620)。
- QES integer update 需要 boundary gate、leaky residual 与 stochastic rounding：[worker_extn_full_precision.py](https://github.com/dibbla/Quantized-Evolution-Strategies/blob/fefc7358decb/utils_int4/worker_extn_full_precision.py#L289)。
- QZO 实际扰动的是量化模块的浮点 `scales`，不是 packed integer qweight：[trainer.py](https://github.com/maifoundations/QZO/blob/ac6803ab2ce7/large_language_models/trainer.py#L1636)。

## 原仓库中抽象后反而更容易看见的问题

1. **seed 不等于可复现。** RandOpt/ES worker 对每个 tensor 新建 generator 并以同一 seed 重置；同形状 tensor 会复用随机流前缀。另一些 reproduction script 却使用 CPU 单一连续 stream。当前 seed JSON 不足以跨实现复现，所以框架把 RNG scheme/version 放进 `NoiseRef`。
2. **反向加法不是精确 reset。** `bf16/fp16: (θ+δ)-δ` 不保证逐 bit 回到 `θ`，长期循环会漂移。backend context 应从 snapshot 恢复；量化 code 还需要保留 evaluation 时的 boundary mask。
3. **parameter scope 不能藏在环境变量。** RandOpt 是否扰动视觉 encoder 由 `PERTURB_VISUAL` 控制，但 artifact 没记录。框架将 scope 变成 replay key 的一部分。
4. **sigma normalization 并不统一。** 有些代码显式除以 sigma，有些让 learning rate 吸收，有些注释掉理论 normalization。`PopulationES(normalization="code" | "score_function")` 刻意把选择暴露出来。
5. **量化方法不是一类。** QZO 更新正的连续 scale；QES 更新有界整数 code，并维护 residual。二者仅共享外层 probe/score 生命周期。
6. **FZOO 当前 parallel prototype 有调试残留。** 因此最小框架只保留 `evaluate_many` 可替换后端，不把特定并行模型改写设计成核心接口。

## 有意没有做什么

- 没有复制或继承 Hugging Face `Trainer`；
- 没有写 torch/JAX tensor materializer；
- 没有 checkpoint、DDP、Ray、scheduler DSL；
- 没有假装所有算法遵循同一个 `1/σ` 或 population normalization；
- 没有把 LOZO/SubZero/learned scaling 拆成过细的 Distribution × Geometry × Scale 类层级；
- 没有将 committee 或 distillation 硬塞成 tensor update。

要把它变成实验框架，下一步最小增量应只是实现一个 torch backend：稳定且去重的 named-parameter traversal、基于参数名 hash 的局部 generator、从 anchor copy 进行 transactional candidate materialization，以及 streaming `Direction` commit。算法控制面无需先改变。
