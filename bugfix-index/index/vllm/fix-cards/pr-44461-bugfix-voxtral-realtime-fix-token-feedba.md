# PR #44461: [Bugfix][Voxtral Realtime] Fix token feedback timeout silent hang

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44461](https://github.com/vllm-project/vllm/pull/44461) |
| 作者 | Sunt-ing |
| 合并日期 | 2026-07-05 |
| labels | bug, mistral, multi-modality, ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

Voxtral realtime 的 feed_tokens 后台任务给队列等待加了迭代超时,空闲间隔误杀任务导致会话静默停止转写;移除该超时修复。

## 问题现象

Voxtral realtime(`/v1/realtime` websocket)会话在经历一段无引擎输出的空闲间隔后,静默停止产出转写:websocket 仍保持连接,后续音频继续到达,但不再有 transcription 增量。两种报告形态:#36015 为单会话在音频间隙后挂起(E2E 复现输出 `REPRO_TIMEOUT_NO_DONE`,收不到 `transcription.done`);#35863 为并发形态——三个并发会话中前两个正常(各 808 个 delta + done),第三个因空闲越过超时,只收到 1 个 delta、无 done、无错误、websocket 一直开着。作者用 `VLLM_ENGINE_ITERATION_TIMEOUT_S=3` 加人为音频间隙把快速 GPU 上约一分钟以上的复现窗口加速到确定性问题。

## 根因

正文明确披露:`feed_tokens()` 后台任务负责把引擎输出 token 从 `input_stream` 搬进实时转写缓冲,它把 `input_stream.get()` 包在 `asyncio.wait_for(..., VLLM_ENGINE_ITERATION_TIMEOUT_S)` 里。引擎没有输出时的正常空闲等待也会触发 `TimeoutError`,把反馈任务直接杀死,而 websocket 会话仍然保持打开——之后音频照常流入,却没有任何生成 token 被追加回实时缓冲,表现为静默挂起。正文并澄清:#36089 的"超时后重试"方案仍把正常空闲当异常,不解决问题;#37483 只改了测试超时/连接校验,未触及生产路径。

## 修复方案

改 `vllm/model_executor/models/voxtral_realtime.py`:移除包在 `input_stream.get()` 外的 `asyncio.wait_for` 及 `VLLM_ENGINE_ITERATION_TIMEOUT_S` 导入,改为直接 `await input_stream.get()`。正文说明:在这个队列上等待下一个引擎输出本身不是失败,等待仍受请求生命周期约束——客户端断连、会话终结与真正的引擎故障仍会通过 finally 块取消/回收生成器,而正常的静默、调度延迟或暂时饥饿的会话不再中断 token 反馈:

```diff
-        # Feed output tokens back into buffer in background
+        # Feed output tokens back into the buffer. Idle waits are normal here;
+        # request cleanup still cancels this task through the finally block.
         async def feed_tokens():
             while True:
-                all_outputs = await asyncio.wait_for(
-                    input_stream.get(),
-                    timeout=VLLM_ENGINE_ITERATION_TIMEOUT_S,
-                )
+                all_outputs = await input_stream.get()
                 await buffer.append_tokens(all_outputs[-1:])
```

E2E 验证:单会话与三并发场景修复后分别输出 `REPRO_DONE` 与 `NO_WEDGE_SESSION3_OK`(三个会话各 808 delta 并收到 `transcription.done`);既有 realtime 校验测试 4 passed、Voxtral realtime 测试 3 passed、pre-commit 通过。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,改动仅删去 `feed_tokens()` 中一处超时包装,影响 Voxtral realtime 的 token 反馈循环:若某处存在预期依赖该超时来终止任务的路径,现在改为依赖请求生命周期/finally 清理来取消。会话终不出即"永远等待"的情形(如清理逻辑失效)会表现为任务常驻而非超时退出。`VLLM_ENGINE_ITERATION_TIMEOUT_S` 在此文件的导入被移除,但环境变量本身仍服务其他路径。其他模型/入口不受影响。

## 关联

- https://github.com/vllm-project/vllm/issues/36015 (Closes)
- https://github.com/vllm-project/vllm/issues/35863 (Closes)
- https://github.com/vllm-project/vllm/issues/36089 (非重复,方案对比)
- https://github.com/vllm-project/vllm/pull/37483 (仅改测试,未触及生产路径)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
