# PR #26394: [PD] Fix cross-rank queue divergence by gating metadata readiness before all-reduce

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26394](https://github.com/sgl-project/sglang/pull/26394) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-05-26 |
| labels | high priority, run-ci |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PD 分离 decode 侧 KV 传输完成不等于元数据落地,poll_and_all_reduce 返回 Success 后各 TP rank 可能对同一请求做出不同的出队决定导致队列分叉;修复为在 all-reduce 前用元数据 gate 把未落地请求降级为 Transferring。

## 问题现象
PD(disaggregation)decode 侧:当 `poll_and_all_reduce` 返回 `KVPoll.Success` 时,仅确认 KV cache 传输完成;经独立 RDMA op 写入的元数据可能尚未在所有 rank 上落地。此时若一个 rank 判定元数据就绪并把请求移出队列,而另一个 rank 仍保留该请求,各 TP rank 的队列便发生分叉。后果是后续 all-reduce 作用在未对齐的 tensor 上,导致挂死或静默数据损坏。

## 根因
正文披露:Success 状态只代表 KV 传输完成、不代表独立写入的元数据已在全部 rank 落地;队列去留决策在各 rank 本地做出且无一致性保障,从而产生 cross-rank queue divergence。

## 修复方案
1. 在 `python/sglang/srt/disaggregation/utils.py` 新增 `_apply_metadata_gate()`:对 poll 值为 Success 的请求,若其 metadata buffer 中 `bootstrap_room` 仍为 0(「尚未写入」哨兵值),则把 poll 从 `Success` 降级为 `Transferring`(fake transfer 除外);在 `poll_and_all_reduce` 与 `poll_and_all_reduce_with_staging` 的单次 all-reduce 之前调用(通过新增可选参数 opt-in)。由于 all-reduce 用 `ReduceOp.MIN`,gate 保证任何 rank 不会在其他 rank 就绪前提交。关键 diff:

```python
+def _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args) -> None:
+    for i, poll_val in enumerate(polls):
+        if poll_val == int(KVPoll.Success):
+            decode_req = decode_reqs[i]
+            if _is_fake_transfer(decode_req.req, server_args):
+                continue
+            actual_room = metadata_buffers.bootstrap_room[
+                decode_req.metadata_buffer_index, 0
+            ].item()
+            if actual_room == 0:
+                polls[i] = int(KVPoll.Transferring)
```

2. 在 `python/sglang/srt/disaggregation/decode.py`:`pop_transferred` 改走带 gate 的 `_poll_with_metadata()` / 带 gate 的 staging 路径;`_commit_transfer_to_req` 不再返回 bool,原先「元数据未就绪则继续留在队列等下一次 poll」的分支改为确定性 abort(正文注释称此时 gate 已在所有 TP rank 确认就绪、不应再发生,若发生则以 INTERNAL_SERVER_ERROR abort 避免跨 rank 队列分叉);同时把 `_is_fake_transfer` 从 decode.py 移到 utils.py 复用。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 PD 分离 decode 侧所有请求的 poll 路径——非调用方(未传新可选参数)行为不变;decode 调度方 `pop_transferred` 每次多读一次 metadata buffer 做 gate 判断;元数据未就绪的处理从「留在队列等下轮」变为「直接 abort 请求」,若 gate 判定与实际落地存在竞态,请求会被 abort 而非重试。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
