# PR #35478: [BugFix] Fix engine hanging after KV cache initialization failure

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35478](https://github.com/vllm-project/vllm/pull/35478) |
| 作者 | 842974287 |
| 合并日期 | 2026-03-06 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 6 |
| 分类 | KV Cache / Memory |
| bug 类型 | hang |

## 一句话结论
KV cache 初始化抛异常时 executor 不被关闭、前端等不到 ready 消息而挂死;现在失败即关 executor,并经 ZMQ 发 FAILED 状态让前端立刻报错。

## 问题现象
vLLM 引擎进程在 KV cache 初始化失败时变成僵尸:engine 进程不退出,前端挂起等待永远不会到达的 ready 消息。按正文测试计划,触发方式例如请求超出可用量的 GPU 显存导致 `_initialize_kv_caches` 抛异常。

## 根因
正文披露:之前若 `_initialize_kv_caches` 抛出异常,executor 从未被 shutdown,前端只能等进程 sentinel 才能发现进程死亡(而 sentinel 可能被清理过程延迟),在握手阶段就挂起——即异常路径缺少 executor 清理与失败通知两条链路。

## 修复方案
三处改动:
1. `vllm/v1/engine/core.py` EngineCore:用 try/except 包住 `_initialize_kv_caches`,失败时记录日志、调用 `self.model_executor.shutdown()` 后再 raise;
2. `_perform_handshake`:yield 期间若发生异常,在 finally 中通过 handshake_socket 发送 `{"status": "FAILED", ...}`,让前端经 ZMQ 立即感知失败而不是等进程 sentinel;正常路径仍发 READY 消息;
3. `vllm/v1/engine/utils.py` `wait_for_engine_startup`:握手响应新增对 `"FAILED"` 状态的处理,直接 raise `RuntimeError("Engine core ... reported initialization failure. See root cause above.")`。

```diff
+        except Exception:
+            logger.exception(
+                "EngineCore failed during KV cache initialization; "
+                "shutting down executor."
+            )
+            self.model_executor.shutdown()
+            raise
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:v1 引擎的启动/握手路径。所有引擎初始化失败场景现在会主动关闭 model_executor 并发送 FAILED 状态,前端以明确的 RuntimeError 快速失败;正常启动路径的 READY 消息逻辑未变,但被重组进 try/finally,DP>1 时附带 parallel_config_hash 的行为保持。对不经过该握手的前端形态,行为差异取决于其是否读取 FAILED 消息。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
