# PR #21519: [Bugfix] Fix incorrect dp-attention parallel info in bench_one_batch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21519](https://github.com/sgl-project/sglang/pull/21519) |
| 作者 | lviy |
| 合并日期 | 2026-04-03 |
| labels | ready-to-merge, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
bench_one_batch 在 DP-Attention 下把 attn_tp_size 硬编码为 1 且漏传 attn_cp_size,DP 准备路径与 scheduler 不一致;改为读取真实并行元数据。

## 问题现象
开启 DP-Attention 时用 bench_one_batch 做低层 profiling(如 `python3 -m sglang.bench_one_batch --model-path .../GLM-4.7-Flash-0309 --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head ... --profile --profile-activities CUDA_PROFILER --profile-stage decode`)会出问题:DP-Attention 准备路径与正常 scheduler 路径不一致,正文以 before/after 截图展示修复前后的差异(修复前 profiling 输出异常)。

## 根因
正文披露:bench_one_batch 本用于低层 profiling,但启用 DP-Attention 时没有把正确的 attention 并行元数据传入 `prepare_mlp_sync_batch_raw()`——`attn_tp_size` 被硬编码为 1,`attn_cp_size` 未转发,使 DP-Attention 准备路径与正常 scheduler 路径不一致。

## 修复方案
`python/sglang/bench_one_batch.py` `_maybe_prepare_mlp_sync_batch`:调用 `prepare_mlp_sync_batch_raw` 时,`attn_tp_size` 由硬编码 `1` 改为 `get_attention_tp_size()`(从 `sglang.srt.layers.dp_attention` 导入),并新增转发 `attn_cp_size=model_runner.attn_cp_size`。

```diff
-            attn_tp_size=1,
+            attn_tp_size=get_attention_tp_size(),
+            attn_cp_size=model_runner.attn_cp_size,
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `bench_one_batch` 的 DP-Attention 同步 batch 准备路径(profiling/基准工具),不涉及正常 scheduler 服务路径;改动为两处参数来源替换,若 `get_attention_tp_size()` 在该调用时机尚未初始化则会暴露新错误(正文未讨论)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
