# PR #43911: [Fix] hisparse: Release HiSparse rows on the KV checksum mismatch and chunked abort exits

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43911](https://github.com/sgl-project/sglang/pull/43911) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-11-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

PD decode KV 校验和失败与 chunked prefill abort 两条退出路径释放 KV 前未释放 HiSparse 状态,造成行泄漏;改为在 `release_kv_cache` 前调用 `hisparse_coordinator.request_finished`。

## 问题现象

启用 HiSparse 时,两个退出路径只调 `release_kv_cache` 释放 KV,却不释放该请求的 HiSparse 状态,导致行泄漏:

- PD decode 的 KV checksum mismatch(`_handle_kv_checksum_mismatch`):此时 transfer 已把该请求 admit 进 HiSparse,其 host pages 与 device buffer 从未被归还;
- chunked prefill abort(`process_pending_chunked_abort`):在 chunk 之间被 abort 的请求只持有其 extend 所分配的 spec scratch row,该行同样无人归还。

泄漏由作者读退出路径代码发现并经单元测试确认;正文说明端到端的服务端复现仍在进行中。

## 根因

正文披露:这两个退出点在用 `release_kv_cache` 释放 KV 时,没有先释放请求的 HiSparse 状态(`request_finished`),因此对应行泄漏。与其他退出路径不同,这两处缺少同一 `enable_hisparse` 守卫下的 `request_finished` 调用。

## 修复方案

在两处退出路径中、`release_kv_cache` 之前,补上与其他出口一致的守卫调用:

```diff
         self.output_streamer.stream_output([req], req.return_logprob)
+        if self.enable_hisparse:
+            self.hisparse_coordinator.request_finished(req)
         release_kv_cache(req, self.tree_cache, checkpoint=False)
```

- `python/sglang/srt/disaggregation/decode.py` 的 `_handle_kv_checksum_mismatch`:`request_finished` 会归还被 admit 进 HiSparse 的 host pages 与 device buffer;
- `python/sglang/srt/managers/scheduler.py` 的 `process_pending_chunked_abort`:走 `request_finished` 的 no-device-buffer 分支,恰好只释放 extend 分配的 spec scratch row,prefill KV 仍由 `release_kv_cache` 释放。

同时为每个退出点各加一个单元测试,断言 HiSparse 释放先于 KV 释放,无修复时两者均失败。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `--enable-hisparse` 路径受影响,未启用时行为无任何变化;未新增或移动任何 stream wait,`request_finished` 保持自己在 forward stream 上的等待,与其他退出点一致。两条路径的释放顺序变为先 HiSparse 后 KV。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
