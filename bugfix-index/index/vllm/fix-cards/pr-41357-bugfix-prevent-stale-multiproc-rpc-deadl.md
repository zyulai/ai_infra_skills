# PR #41357: [Bugfix] Prevent stale multiproc RPC deadlines from becoming unbounded waits

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41357](https://github.com/vllm-project/vllm/pull/41357) |
| 作者 | bugkeep |
| 合并日期 | 2026-07-29 |
| labels | bug, ready, v1 |
| 热度 | 👍 4 · 💬 5 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
multiproc RPC 过期 deadline 以负数传给 zmq poll 成为无界等待,钳制剩余超时为 0 并在 recv 层加兜底防护修复。

## 问题现象
`MultiprocExecutor.collective_rpc()` 场景下:同步 RPC 先排空积压的异步 future 时会流逝时间,使已捕获的绝对 deadline 在读取当前调用响应前过期,剩余超时算出负值。负 float 一路传到 `MessageQueue.dequeue()` → `MessageQueue.recv()` → `zmq.Socket.poll(timeout=<负数>)`,ZMQ 把负值视为未定义行为(常表现为无界等待),即 RPC 层一类可表现为无界挂起的超时 bug。正文明确说明它修复的是与 #40926(slow KV connector waits)相关的一类问题,不声称是 #40926 的完整修复。

## 根因
正文披露根因:`collective_rpc()` 在入队时计算绝对 deadline(`deadline = time.monotonic() + timeout`),`get_response()` 闭包捕获该 deadline。当同步 RPC 先排空积压的异步 future,时间流逝,deadline 可能在读当前响应前过期;此前剩余时间按普通减法计算(`dequeue_timeout = deadline - time.monotonic()`),可能为负。负值到达 `zmq.Socket.poll(timeout=...)` 后语义未定义,往往变成无界等待。

## 修复方案
1. 新增 `_remaining_timeout()` 辅助函数:钳制为 `max(0.0, deadline - time.monotonic())`,过期 deadline 产生立即超时(0.0)而非负值。
2. 同步 RPC 绕过 `FutureWrapper`:`non_block=False` 时直接排空积压异步 future 再调 `get_response()`,避免无谓分配,语义不变(相同排空顺序、聚合、异常传播)。
3. `MessageQueue.recv()` 兜底防护:毫秒超时钳制为 `max(0, ...)`,即使未来代码路径绕过 `_remaining_timeout`,负值也到不了 `zmq.Socket.poll()`。
4. 新增 25 个单测(两个文件):`test_rpc_deadline_fix.py`(16 个)与 `test_multiproc_executor_timeout.py`(9 个,用 `FakeClock` 时间旅行,主回归测试 `test_future_wrapper_stale_deadline_never_passes_negative_timeout` 断言 `FakeResponseMQ.timeouts == [0.0]` 且 dequeue 永不收到负值)。
明确不变项:公共 API 不变、`non_block=True` 路径不变、排空顺序(FIFO)不变、`KVOutputAggregator`/`unique_reply_rank` 聚合语义不变、异常传播不变。

关键 diff 摘录:
```diff
-        timeout_ms = None if timeout is None else int(timeout * 1000)
+        # Ensure non-negative timeout passed to zmq poll.
+        timeout_ms = None if timeout is None else max(0, int(timeout * 1000))
...
                 dequeue_timeout = (
-                    None if deadline is None else (deadline - time.monotonic())
+                    None if deadline is None else max(0.0, deadline - time.monotonic())
                 )
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:`vllm/v1/executor/multiproc_executor.py` 的 `get_response()` 剩余超时计算、`vllm/distributed/device_communicators/shm_broadcast.py` 的 `recv()`,以及 multiproc RPC 同步路径的 future 包装方式。过期 deadline 的行为从「未定义/无界等待」变为「立即 TimeoutError」;未过期 deadline、None deadline、聚合与异常传播语义均有单测锁定。

## 关联
- [#40926](https://github.com/vllm-project/vllm/pull/40926)(正文 Related 节引用:slow KV connector waits,本 PR 仅修其中一类超时 bug)
- #41213(正文提及本 PR 取代的前置 PR,仅做最小 inline clamp;正文未给链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
