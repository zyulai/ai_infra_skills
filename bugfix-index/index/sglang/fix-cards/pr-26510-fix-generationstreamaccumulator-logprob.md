# PR #26510: Fix _GenerationStreamAccumulator logprob_end off-by-one under retract

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26510](https://github.com/sgl-project/sglang/pull/26510) |
| 作者 | shenxiul |
| 合并日期 | 2026-08-20 |
| labels | run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
overlap 调度下请求在 prefill 结果处理前被 retract,output logprob 游标差一;retract 时去掉 max(...,1) 下限,游标保持 0。

## 问题现象
overlap 调度 + retraction 场景下,`return_logprob=True` 的非流式响应输出 logprob 数量比 output token 少一个,且后续每次输出都错位一个 logprob。触发条件:请求的 prefill forward 已发出、结果尚未处理时该请求被 retract(`req.is_retracted = True`)。正文给出的端到端复现:修复前连续 5 次试验中有 2/32 响应错配,修复后连续 8 次试验 0/32。回归测试用 `SGLANG_TEST_RETRACT=True`(每两个 forward step 强制 retract 一次)稳定触发,修复前约 6% 的响应少一条 logprob。

## 根因
正文明确披露了五步链条:

1. prefill forward pass 发出;
2. 该 pass 在途时请求被 retract,`req.is_retracted = True`;
3. `process_batch_result_prefill` 跳过为该请求 append 采样 token,但仍把整个 batch 传给 output streamer;
4. 对没有 output IDs 的非流式请求,强制流式条件为真,因为 `0 % DEFAULT_FORCE_STREAM_INTERVAL == 0`;
5. `max(len(output_ids_), 1)` 把 `send_output_token_logprobs_offset` 推进到 1,而 `send_token_offset` 仍为 0。

两个发送游标从此相差 1:该请求后续每次流式送出 N 个 token 却只带 N-1 条 logprob,最终响应满足 `len(meta_info["output_token_logprobs"]) == len(output_ids) - 1`。

## 修复方案
改 `python/sglang/srt/managers/scheduler_components/output_streamer.py` 的 `_GenerationStreamAccumulator.accept`:对 retract 的请求去掉 `max(..., 1)` 下限,两个游标都停在 0;非 retract 的 prefill-only(`max_new_tokens=0`)请求保留原有「返回第一条 logprob」的行为,修复刻意只针对 retract 场景。

```diff
             if req.return_logprob:
-                logprob_end = max(len(output_ids_), 1)
+                logprob_end = (
+                    len(output_ids_) if req.is_retracted else max(len(output_ids_), 1)
+                )
```

测试:新增 CPU 单测(retract 空输出请求不推进 logprob 游标;非 retract `max_new_tokens=0` 请求仍返回第一条 logprob),并把端到端强制 retract 回归测试注册进 CUDA `base-b` 与 AMD `stage-b-test-1-gpu-small-amd` CI。

## 影响与回归风险
- 回归面:仅影响 `output_streamer.py` 中 `return_logprob` 请求的 logprob 切片游标计算;`req.is_retracted` 为 True 且输出为空时游标不再前进,其余路径行为不变。测试用 `SGLANG_TEST_RETRACT` 环境变量强制 retract 触发,属低频路径。

## 关联
正文未列出(正文仅 cc @Qiaolin-Yu)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
