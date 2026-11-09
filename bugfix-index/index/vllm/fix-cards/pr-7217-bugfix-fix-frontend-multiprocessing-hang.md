# PR #7217: [BugFix] Fix frontend multiprocessing hang

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7217](https://github.com/vllm-project/vllm/pull/7217) |
| 作者 | maxdebayser |
| 合并日期 | 2024-08-07 |
| labels | ready |
| 热度 | 👍 2 · 💬 10 |
| 分类 | Frontend / Serving |
| bug 类型 | hang |

## 一句话结论

多进程模式下后端进程初始化即死会让前端永久挂起;给就绪 RPC 加超时并检测进程存活,改为快速报错。

## 问题现象

自 OpenAI server 支持可选多进程模式(PR #6883)后,如果后端(引擎)进程在初始化阶段就崩溃,它永远不会回复 `async_engine_client.setup()` 中发出的 `IS_SERVER_READY` 消息,前端随之无限等待,表现为启动 hang。触发条件是"多进程模式 + 后端初始化期间死亡",凡初始化失败(如引擎构造抛异常)的场景都会命中;正文未给出具体日志或版本号。

## 根因

正文披露:`_send_one_way_rpc_request` 发出 `IS_SERVER_READY` 后等待应答时没有任何超时机制,后端进程在初始化期间死亡后不会作出应答,客户端便无限阻塞在等待应答上,造成 hang。另外,diff 中 socket 关闭处补上 `linger=0` 的注释说明:不设 linger 时 `context.destroy()` 会为未发送消息等待约 30 秒,而服务端已崩溃时这些消息永远不可能被接收(由 diff 推断)。

## 修复方案

改动三处:

1. `vllm/entrypoints/openai/rpc/client.py`:`_send_one_way_rpc_request` 新增可选 `timeout` 参数,发出请求后用 `socket.poll(timeout=timeout)` 等待应答,超时抛 `TimeoutError(f"server didn't reply within {timeout} ms")`;新增常量 `SERVER_START_TIMEOUT_MS = 1000`,`wait_for_server()` 发送 `IS_SERVER_READY` 时传入该超时。同时 socket 上下文管理器把 `socket.close()` 改为 `socket.close(linger=0)`,关闭即丢弃未发送消息,避免服务端崩溃后 `context.destroy()` 长时间等待。
2. `vllm/entrypoints/openai/api_server.py`:`build_async_engine_client` 中把 `setup()` 移入重试循环,捕获 `TimeoutError` 后检查 `rpc_server_process.is_alive()`,进程已死则抛 `RuntimeError` 快速失败:

```diff
+            while True:
+                try:
+                    await async_engine_client.setup()
+                    break
+                except TimeoutError as e:
+                    if not rpc_server_process.is_alive():
+                        raise RuntimeError(
+                            "The server process died before "
+                            "responding to the readiness probe") from e
```

3. 新增 `tests/entrypoints/openai/test_mp_crash.py`:monkeypatch `AsyncLLMEngine.from_engine_args` 使其抛异常,断言 `build_async_engine_client` 抛出的 `RuntimeError` 含 "The server process died before responding to the readiness probe"。

## 影响与回归风险

- 影响窗口:正文只说明问题自 #6883 引入可选多进程模式后存在,未给出具体版本/提交范围。
- 回归面:按 diff 实际改动,`_send_one_way_rpc_request` 的 `timeout` 默认为 `None`,其余 one-way RPC 调用路径行为不变;只有 `wait_for_server()` 的就绪探测改为 1 秒轮询——若进程存活但应答偏慢,会反复超时重试直到启动成功或进程死亡。`socket.close(linger=0)` 在关闭时丢弃未发送消息,按 diff 注释,正常路径下应答总是在关闭前收到,故主要影响的是崩溃场景。`api_server.py` 启动路径由"无限挂起"变为"进程死亡时抛 RuntimeError",错误语义变化可能影响依赖旧挂起行为的调用方。

## 关联

- https://github.com/vllm-project/vllm/issues/7213 (FIX #7213)
- https://github.com/vllm-project/vllm/pull/6883 (可选多进程模式来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
