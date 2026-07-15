# noise_floor 执行进度

- 状态：预注册 GPU 测量与验收已完成；所有硬判据通过，Part 6c 因条件不满足跳过。
- 执行依据：`EXPERIMENT_PLAN.md`（2026-07-12）。
- S0-1 PASS；S1-1 PASS；S1-3 PASS。S2-1、P6-1、P6-2 为软判据/预测 FAIL，均已如实记录。
- Part 6c：跳过（σ* ≤ 1e-6，不在 `[1e-4,1e-2]`）。
- 正式 GPU 单元全部一次完成；无 OOM、无降配、无 600 秒退避；仅用 GPU 0。
