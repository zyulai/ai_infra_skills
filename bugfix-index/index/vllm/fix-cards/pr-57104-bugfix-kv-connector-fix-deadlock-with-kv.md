# PR #57104: [BugFix][KV Connector] Fix Deadlock with KVConnector + MTP under KV Pressure

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57104](https://github.com/vllm-project/vllm/pull/57104) |
| 作者 | robertgshaw2-redhat |
| 合并日期 | 2026-09-16 |
| labels | bug, kv-connector, needs-rebase, ready, scheduler |
| 热度 | 👍 0 · 💬 15 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
MTP + KV connector 在 KV 压力下,async load 升级时 MTP 块分配失败并永久 break 循环导致死锁;准入时为 spec decode step 预留块。

## 问题现象
Scheduler 在 KV 内存压力下配合 MTP + KV connector 会死锁(生产部署 GLM5.2 中遇到)。正文给出的例子:总 7 块,A 需 4+1(MTP)块、B 需 3+1 块。两个请求都被准入 `skipped_waiting` 并置 `WAITING_FOR_REMOTE_KVS`;A 的传输到达后被提升为 WAITING,但调度 A 时还要为 MTP 分配 1 块(async load 之后才分配),此时 0 块可用,`allocate_slots` 返回 None 并 `break`;A 一直卡在队首,状态既非 blocked-waiting 也不推进,B 永远轮不到;若无请求在跑,就没有块被释放,循环永久空转。

## 根因
正文披露:async KV load 分配时不带 lookahead slots(`limit_lookahead_tokens`),而支撑 `_inflight_prefill_reserved_blocks()` 准入门控的 `_request_remaining_blocks()` 也不计这些块;待 `WAITING_FOR_REMOTE_KV` 完成后再向 `allocate_slots()` 要 lookahead margin,拿不到就 `break`——预留块数与实际开跑所需块数不匹配,是死锁根源。

## 修复方案
`vllm/v1/core/sched/scheduler.py` 新增 `_spec_decode_step_blocks()`:

```diff
+    def _spec_decode_step_blocks(self) -> int:
+        if not self.num_spec_tokens:
+            return 0
+        return cdiv(
+            1 + self.num_spec_tokens + self.num_lookahead_tokens, self.block_size
+        )
```

两处接线:async load 准入时 `reserved_blocks = self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks()`;`_request_remaining_blocks()` 返回值也加上 `_spec_decode_step_blocks()`(docstring 改为 "hold its full sequence and be promoted")。新增单测 `test_async_load_reserves_blocks_for_promotion_margin`(7 可用块、两请求恰好填满时,B 必须被留在 WAITING 不占块,保证 A 可提升并真正调度)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;触发条件是 KVConnector async load + spec decode(MTP)+ KV 池被打满。
- 回归面:带 spec decode 时的 async KV load 准入与 `_request_remaining_blocks()` 准入门控——准入会更保守(多预留 `cdiv(1+num_spec_tokens+lookahead, block_size)` 块),可能推迟部分请求准入;非 spec decode 路径返回 0,行为不变。

## 关联
- [reproducer 评论](https://github.com/vllm-project/vllm/pull/57104#issuecomment-5701283900)(正文 Test Plan 提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
