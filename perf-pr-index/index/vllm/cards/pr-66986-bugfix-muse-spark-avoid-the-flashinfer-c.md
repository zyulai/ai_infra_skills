# PR #66986: [Bugfix][Muse Spark] Avoid the FlashInfer Cake PDL race in gated_residual_rmsnorm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-11-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | bug, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66986 |

## 一句话结论

Muse Spark 的 gated_residual_rmsnorm 改走 FlashInfer CuTe DSL backend,修复 DSpark DEP8 的 NaN 输出。

## 动机

Muse Spark 的 `gated_residual_rmsnorm` 默认走 FlashInfer Cake backend。Cake 的 programs 在 ≤128 rows 时于 `griddepcontrol.wait` 之前就读取 `branch`(flashinfer-ai/flashinfer#9343),破坏 DSpark 读取的 hidden states,使 DP+EP、TP=1 的 DSpark 输出 garbage(DEP8 AIME25 0%)。当时无 open PR 修复此问题。

## 优化点

根因链:DSpark 经 `norm_then_gated_residual` 读 5 个 auxiliary target hidden states,该调用以 `backend="auto"`、`enable_pdl=True` 走 `flashinfer.norm.gated_residual_rmsnorm`;1–128 rows 时 Cake PDL programs 预取到 stale `branch` 产生 NaN,经 draft 的 context K/V 进入 logits——greedy 下多数 draft 被拒(sdAL≈1)但被接受的 NaN draft 破坏输出(GSM8K 23%),sampling 下 NaN draft 概率总通过验收,输出全 garbage(AIME25 0%,sdAL 8.00)。修法:`fused_rmsnorm_fwd` 显式传 `backend="cute"`,`supports_gated_residual_rmsnorm` 也以 `backend="cute"` 询问,使 CuTe 跑不了的形状回退既有 Triton path 而非报错。不用 `enable_pdl=False` 的原因:两者端到端打平,CuTe 保留 PDL 且 decode-size 调用约快 0.5 µs;关 PDL 在 ≥2048 rows 的 prefill chunk 每调用快 7–13%,但只占 140k prefill 的 0.05% 以下。只有 TP=1 触发:trigger 依赖它前面跑的是哪个 kernel;target 自身的边界用无 PDL 的 `gated_residual_next_rmsnorm`,故 target-only 不受影响。Revert 计划:FlashInfer 修复后回到默认 backend。

## 关键改动

```text
vllm/models/muse_spark/common/ops/fused_rmsnorm.py: gated_residual_rmsnorm 调用与 supports 检查均显式 backend="cute",附指向 flashinfer#9343 的注释
```

## 性能声明

B300,Muse Spark NVFP4 + DSpark(7 draft tokens),FP8 KV。

- 修复效果 DEP8:AIME25(3 epochs,temp 0.6)0%→98.9%(pass@3 100%);GSM8K 5-shot strict/flexible(greedy)23.1%/27.2%→94.8%/94.8%;sdAL AIME25 8.00→3.66、GSM8K 1.01→3.91。target-only DEP8 AIME25 100%;TEP4 GSM8K 95.0%(sdAL 3.90)。
- TEP4 TPOT(140k in / 512 out,95% prefix hit,3 次取中位):c1 3.61→3.61 ms、c8 8.27→8.73 ms、c48 19.16→18.84 ms;c=8 差异为 acceptance noise(sdAL 2.26 vs 2.33;每步时间 19.66 vs 19.41 ms);TTFT 各并发差 <1%,TEP4 端到端无 perf 变化。
- Per-call latency(B300,CUDA graph of 64 chained calls;kernel 每 decode step 跑 5 次,每个 captured aux layer 一次):rows 8:1.50→1.58 µs;48:2.06→2.14 µs;384:6.38→6.20 µs;4096:52.9→55.1 µs。

## 验证与回归风险

无新增单测:race 需要 producer 提前触发 dependents,issue 提供 standalone repro;作者以端到端 DEP8 eval 覆盖。命令:pre-commit、`lm_eval --tasks gsm8k --num_fewshot 5`、`mqa-run --task aime25`。风险:依赖 FlashInfer CuTe backend 可用性,不支持的形状回退 Triton;flashinfer#9343 修复后即可 revert 回 Cake。

## 关联

flashinfer-ai/flashinfer#9343(上游 bug 与修复跟踪);pack 注明无其他 open PR 处理此问题。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
