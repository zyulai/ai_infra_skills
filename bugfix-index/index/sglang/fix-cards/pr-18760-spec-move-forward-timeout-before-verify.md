# PR #18760: [Spec] Move forward timeout before verify to fix Eagle v1 filter mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18760](https://github.com/sgl-project/sglang/pull/18760) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-02-13 |
| labels | high priority, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Eagle v1 下 forward 超时在 verify 之后检查,额外结束请求与 topk_p 已过滤长度不一致报 ValueError;修复为把超时检测移到 batch 发射前,只设 to_finish 交由 verify 的 check_finished 转换。

## 问题现象
Eagle speculative decoding v1 下报 `ValueError: length of new_indices != length of topk_p`(filter 长度不匹配)。触发条件:启用 `SGLANG_FORWARD_TIMEOUT_MS` 的 forward 超时检测,且超时请求在 spec-v1 verify 之后才被处理。相关:#14742、#17831。

## 根因
正文披露:forward timeout 原本在 `process_batch_result_decode` 中检查,而它运行在 verify 已经把 `topk_p` 过滤成 N-K 条之后。超时设置了 `req.to_finish`,`check_finished()` 立即将其转换为 `finished_reason`,导致额外 M 个请求被标记结束;下一轮 `filter_batch(v1_spec_info_filtered=True)` 发现 N-K-M 个活跃请求,但 `topk_p` 仍是 N-K 条——长度不匹配。abort 没有这个问题,因为它只在 `process_input_requests` 中设置 `to_finish`,`filter_batch` 只检查 `finished_reason`,挂起的 `to_finish` 由下一轮 forward 中 verify 的 `check_finished()` 捕获,与 `topk_p` 保持一致。

## 修复方案
按正文 Fix 部分:
1. `python/sglang/srt/managers/scheduler.py`:新增 `_check_forward_timeout_for_running_batch()`,并在 `get_next_batch_to_run()` 里紧跟 `_abort_on_queued_timeout()` 之后调用——超时检测从 batch 结果处理阶段移到 batch 发射之前,复刻 abort 的模式(只设 `to_finish = FINISH_ABORT("Forward timeout.", SERVICE_UNAVAILABLE)`,由 verify 的 `check_finished()` 完成转换)。关键 diff:

```python
+    def _check_forward_timeout_for_running_batch(self):
+        # NOTE: this should be called before a batch is launched,
+        # as current spec-v1 still filters batch inside verify stage.
+        timeout_ms = envs.SGLANG_FORWARD_TIMEOUT_MS.get()
+        if timeout_ms <= 0:
+            return
+        if self.running_batch.is_empty():
+            return
+        deadline = time.perf_counter() - timeout_ms / 1000.0
+        for req in self.running_batch.reqs:
+            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
+                req.to_finish = FINISH_ABORT(
+                    "Forward timeout.", HTTPStatus.SERVICE_UNAVAILABLE
+                )
```

2. `scheduler_output_processor_mixin.py`:从 `process_batch_result_decode` / `process_batch_result_prefill` 中删除原超时检查(deadline 计算与 `req.to_finish` 赋值)。
3. `multi_layer_eagle_worker.py` 的 `forward_draft_extend`:删除死代码过滤分支——该处 `req.finished()` 恒为 False,因为在那之前从未调用过 `check_finished()`。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响所有启用 `SGLANG_FORWARD_TIMEOUT_MS` 的调度路径:超时请求的结束时机从「结果处理阶段立即 finish」推迟到「下一轮 forward 的 verify check_finished」,非 spec 模式下请求会在下一次 batch 发射前被标记 to_finish,取消延迟略增;未启用超时(<=0)时新函数直接返回,行为不变;`forward_draft_extend` 删除的过滤分支正文确认为死代码。

## 关联
- [#14742](https://github.com/sgl-project/sglang/pull/14742)(相关)
- [#17831](https://github.com/sgl-project/sglang/pull/17831)(相关)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
