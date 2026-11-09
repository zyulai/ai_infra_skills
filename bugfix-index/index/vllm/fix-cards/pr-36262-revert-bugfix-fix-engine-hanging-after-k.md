# PR #36262: Revert "[BugFix] Fix engine hanging after KV cache initialization fai…

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36262](https://github.com/vllm-project/vllm/pull/36262) |
| 作者 | njhill |
| 合并日期 | 2026-03-06 |
| labels | bug, v1 |
| 热度 | 👍 0 · 💬 0 |
| 分类 | KV Cache / Memory |
| bug 类型 | hang |

## 一句话结论
回退 #35478「KV cache 初始化失败后修复 engine 挂起」的改动:其引入测试失败,且希望简化核心逻辑改动,整体撤销。

## 问题现象
正文未描述具体现象。本 PR 是对 #35478 的回退,回退理由在正文中披露:被回退的改动「introduced test failures(引入了测试失败)」,并且维护者希望简化对核心逻辑的改动。

## 根因
正文未披露根因。正文仅给出回退动机:被回退 PR(#35478,commit 0a208d1f549a5e35605af5b01685d64cd727b73b)带来了测试失败,且其对核心逻辑的改动需要简化。由 diff 推断:被回退的改动在 `vllm/v1/engine/core.py` 中为 KV cache 初始化加了 try/except(失败时关闭 model_executor 并向上抛)、并在 startup 握手中增加 "FAILED" 状态消息发送;回退将这些全部撤掉,恢复原有逻辑。

## 修复方案
按 revert 原样撤销 #35478 的三处改动:

1. `vllm/v1/engine/core.py`:移除 `_initialize_kv_caches` 外层的 try/except(原先失败时记录日志、调用 `self.model_executor.shutdown()` 再 raise),恢复为直接赋值:
```diff
-        try:
-            num_gpu_blocks, num_cpu_blocks, kv_cache_config = (
-                self._initialize_kv_caches(vllm_config)
-            )
-        except Exception:
-            ...
-            self.model_executor.shutdown()
-            raise
+        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
+            vllm_config
+        )
```
2. `vllm/v1/engine/core.py` `_perform_handshake`:移除 `exc_during_init` 分支(初始化失败时通过握手 socket 发送 `status: "FAILED"` 让前端经 ZMQ 立即感知),恢复为仅 `yield addresses` 后发送 READY 消息。
3. `vllm/v1/engine/utils.py` `wait_for_engine_startup`:移除对 `status == "FAILED"` 消息的处理分支(原先会 raise "Engine core ... reported initialization failure")。

## 影响与回归风险
- 影响窗口:从引入 #35478 的提交起至本回退合并前,包含该提交的版本受影响。
- 回归面:回退后 KV cache 初始化失败时不再有 executor 主动 shutdown 与握手 FAILED 快速失败路径,前端对初始化失败的感知退回原机制(等待进程 sentinel);原 #35478 针对的 engine 挂起场景可能重新出现。改动均集中在 `vllm/v1/engine/core.py` 与 `vllm/v1/engine/utils.py`。

## 关联
[#35478](https://github.com/vllm-project/vllm/pull/35478)(正文以 revert 标题形式引用,commit 0a208d1f549a5e35605af5b01685d64cd727b73b)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
