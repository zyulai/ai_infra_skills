# PR #45448: [Bugfix] Complete one-shot fused all-reduce PDL at end to avoid NaN

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45448](https://github.com/vllm-project/vllm/pull/45448) |
| 作者 | alexeldeib |
| 合并日期 | 2026-06-18 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

one-shot Lamport all-reduce 提前上报 PDL 完成,后续 kernel 读到未初始化缓冲产生 NaN;one-shot 路径改为末尾完成。

## 问题现象

在 TP>1 且启用 FlashInfer trtllm fusion 的部署下(复现配置:`vllm serve <model> --tensor-parallel-size 4` + eagle3 MTP 投机解码 `{"method":"eagle3","num_speculative_tokens":4}` + `--compilation-config '{"mode":3,"pass_config":{"fuse_allreduce_rms":true}}'`,例如 Kimi-K2.6),concurrency=1 时每个 decode step 的输出都是 NaN,表现为响应内容为空:`"content": "", "finish_reason": "length", completion_tokens: 40`。修复前的临时规避是 `fuse_allreduce_rms: false`;修复后 "no more NaN/all decode succeeds"。

## 根因

PR 正文披露:FlashInfer one-shot Lamport all-reduce 在 `trigger_completion_at_end=False` 时,会在输出 buffer 尚未提交(commit)前就上报 PDL(programmatic-dependent-launch)完成,下一个 PDL 启动的 kernel 由此读到未初始化的 Lamport buffer,产生 NaN。融合 AR+RMSNorm pass 在 #43103 中把该 flag 设为 `num_tokens <= PDL_ADVANCE_LAUNCH_TOKENS`——这恰好是 batch=1 / spec-decode 形状,而 one-shot 路径总在这些形状下被选中,于是张量并行下每个 decode step 都 NaN。

## 修复方案

改 `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` 的 `call_trtllm_fused_allreduce_norm`:以 `use_oneshot` 作为门控——one-shot 路径强制 `trigger_completion_at_end=True`(在末尾完成 PDL),two-shot 路径内部已同步,保留 early completion 及其性能收益:

```diff
-            trigger_completion_at_end=num_tokens > PDL_ADVANCE_LAUNCH_TOKENS,
+            trigger_completion_at_end=use_oneshot
+            or num_tokens > PDL_ADVANCE_LAUNCH_TOKENS,
```

正文说明:独立通信器 `flashinfer_all_reduce.py` 共享同一 kernel(`launch_with_pdl=True`),需要同样修复,但其 workspace 大小无法服务大 hidden 模型(推理前校验即失败),留作后续 PR。

## 影响与回归风险

- 影响窗口:由 #43103 引入该 flag 逻辑后的版本受影响(正文口径);症状出现在 batch=1 / spec-decode 形状。
- 回归面:仅 `call_trtllm_fused_allreduce_norm` 的 PDL 完成时机;one-shot 路径失去 early-completion 的性能优化(正确性优先),two-shot 路径行为与性能不变。独立 flashinfer all-reduce 路径此次未改。

## 关联

- https://github.com/vllm-project/vllm/pull/43103
- https://github.com/vllm-project/vllm/pull/40557
- https://github.com/flashinfer-ai/flashinfer/issues/1223

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
