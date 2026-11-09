# PR #18958: [spec v2] Fix torch gc of future indices

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18958](https://github.com/sgl-project/sglang/pull/18958) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-02-19 |
| labels | high priority, run-ci |
| 热度 | 👍 9 · 💬 3 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论

spec v2 中 FutureMap 的 indices 张量在默认流分配、forward 流使用且引用提前归零,被 torch 缓存分配器提前回收引发 IMA;加 `record_stream` 修复。

## 问题现象

spec v2(投机解码 v2)路径出现 IMA。作者给出在小模型上的复现方法:llama3-8b on H200,在 `scheduler.py` 的 forward 执行段 `self.forward_stream_ctx` 内、`resolve_future` 之前插入 `torch.cuda._sleep(1_000_000_000)` 即可复现(拉长 forward 流执行时间,放大窗口)。关联 issue #18744。

## 根因

正文详细披露了排查与根因:作者认真分析过 nvcastet 与 trevor-m 提出的数据竞争猜想,结论是并不存在传统意义的数据竞争——`prepare_for_decode` 与 `_draft_extend_for_decode` 虽同时访问共享 buffer `req_to_token`,但两阶段之间没有冲突,即使有也不会导致越界或 IMA。随后转向「张量因未跨流 record 而被垃圾回收」的假设:有些张量在 scheduler(默认)流上创建、却在 forward_stream 上使用,且其引用计数在 forward 期间就降为零。作者先对 ModelWorkerBatch 里所有 GPU 张量做整体 clone 做消融,IMA 依旧;trevor-m 提示 FutureMap 里的 indices 值不合理,最终定位根因——`future_indices.indices` 在默认流上分配、在 forward_stream 上使用,而其 Python 引用在 GPU 读完之前就被丢弃(`model_worker_batch.spec_info` 与 `batch.spec_info` 被替换时),缓存分配器(torch GC)可能提前回收该内存。

## 修复方案

改 `python/sglang/srt/managers/overlap_utils.py` 的 `resolve_future`:在使用 indices 之前调用 `record_stream`,告知分配器该张量还会被当前(forward)流使用:

```diff
             indices = draft_input.future_indices.indices
+            # The indices tensor was allocated on the default stream but is
+            # used here on the forward stream. ... the caching allocator
+            # (torch GC) could reclaim the memory before the GPU finishes
+            # reading it.
+            indices.record_stream(torch.get_device_module(self.device).current_stream())
             draft_input.topk_p = self.topk_p_buf[indices]
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围;受影响路径为 spec v2 的 `FutureMap.resolve_future`(overlap_utils),未加 record_stream 的版本存在 IMA 风险。
- 回归面:改动为单行 `record_stream` 调用,仅延迟该 indices 张量的内存回收,不影响数值与控制流;`record_stream` 会稍微延后缓存复用,理论上有轻微内存占用影响。非 spec v2 路径(提前 return 的 prefill batch 分支)不受影响。

## 关联

- [#18744](https://github.com/sgl-project/sglang/issues/18744)(正文 "fix" 引用的 issue)
- [#18803](https://github.com/sgl-project/sglang/pull/18803)(正文 "close" 引用的 PR)
- 排查中 @nvcastet、@trevor-m 提供的分析与提示

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
