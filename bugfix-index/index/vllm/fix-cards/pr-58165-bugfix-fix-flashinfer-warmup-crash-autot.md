# PR #58165: [Bugfix]Fix FlashInfer warmup crash: autotune(tuning_buckets=...) must round M up, not down (Invalid MXFP8 split-K tactic with spec-decode)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58165](https://github.com/vllm-project/vllm/pull/58165) |
| 作者 | ebfio |
| 合并日期 | 2026-10-07 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
warmup 的 `autotune(tuning_buckets=...)` 默认向下取 M 到 bucket,选中无效 MXFP8 split-K tactic 导致带 spec-decode 的启动崩溃循环;改为 `round_up=True` 并把 drafter 的 M 纳入 bucket。

## 问题现象
SM100/SM103 上 block-scaled cute-dsl 线性 kernel(`mm_mxfp8` / `mm_fp4`)在开启投机解码(spec-decode)时启动即崩溃循环:engine 起不来。`kernel_warmup` 在 `fi_utils.autotune(tuning_buckets=...)` 内跑 dummy forward,spec-decode 下 DSpark draft model 前向以 M = `max_num_seqs × k`(如 8×3=**24**,不是 bucket 值)运行,启动报:

```
ValueError: Invalid MXFP8 split-K tactic: ((128, 16), (1, 1), True, False, 4)
```

出错位置 `flashinfer/gemm/gemm_base.py:5327`(`CuteDSLMxfp8GemmRunner.forward`),由 draft model 的 `fused_wqa_wkv` GEMM 到达。在 DGX Station GB300(SM103)、vLLM nightly + DeepSeek-V4.1-Flash + dspark k=3 复现;FlashInfer 单独复现见上游 issue。

## 根因
正文披露:`mm_mxfp8` 自身 tuning config 用 `map_to_hybrid_bucket_uncapped` 把运行时 M **向上**取整到 bucket,cute-dsl split-K MMA tile 也按向上取整选择(`mma_tiler_mn_for_m`: 8/16/32),故 bucket b 处调好的 tactic 对所有 M ≤ b 有效。但 `autotune(tuning_buckets=...)` 会把 op 的 mapper 替换为 `make_bucket_mapper(buckets, round_map=round_up)` 且 `round_up` 默认 False(floor):warmup 的 M=24 命中 **bucket-16** tactic(tile (128,16), split_k=4),而 `forward()` 校验 `mma_tiler_mn_for_m(24)` 应为 (128,32),拒绝之。context 外(serve、CUDA graph capture)恢复 op 自带向上 mapper,所以只有 warmup 必崩——engine 永远起不来,而同样权重/tactic/cache 正常 serving 没问题。正文另提示更广隐患:swap-AB tile 的 `can_implement` 要求 `kernel_n ≤ tile_n` 且无 apply-time 检查,floor 映射下会静默越界;`mm_fp4` 共用同一 tactic 生成器,NVFP4 + spec-decode 同样暴露。

## 修复方案
`vllm/model_executor/warmup/kernel_warmup.py` `_run_flashinfer_autotune_dummy_runs`:
- warmup 调用改为 `with fi_utils.autotune(tuning_buckets=tuning_buckets, round_up=True):`,使 tune 期映射与 serving 期一致;

```diff
-        with fi_utils.autotune(tuning_buckets=tuning_buckets):
+        with fi_utils.autotune(tuning_buckets=tuning_buckets, round_up=True):
```

- 构造 bucket 时纳入 drafter 的最大 M:`speculator.num_query_per_req` × `num_draft_reqs`(取 `min(num_tokens, max_num_reqs, max_num_tokens // num_query_per_req)`),`max_tuning_tokens = max(num_tokens, num_draft_reqs * num_query_per_req)`。
Profiling 按 bucket 进行,tactic 选择与 cache 文件不变,仅 dummy run 执行路径变化。GB300 实测:同配置原先崩溃循环的启动恢复干净、可带完整 CUDA graph 服务,既有 cache 上 warmup 通过。

## 影响与回归风险
- 影响窗口:正文未说明(复现环境为 vLLM nightly + SM103,FlashInfer 侧 issue #5450)。
- 回归面:仅 FlashInfer kernel warmup 的 autotune 上下文:round_up=True 后每 bucket 的 tactic 只会被应用于 M ≤ b,与 split-K 校验器及小 tile swap-AB 的适用包络一致;cache 文件不变;serving/CUDA graph capture 路径不受影响。

## 关联
- 上游分析 https://github.com/flashinfer-ai/flashinfer/issues/5450

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
