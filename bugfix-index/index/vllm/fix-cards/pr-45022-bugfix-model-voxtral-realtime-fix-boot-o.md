# PR #45022: [Bugfix][Model] Voxtral realtime: fix boot OOM and engine crash at max_model_len

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45022](https://github.com/vllm-project/vllm/pull/45022) |
| 作者 | damienlaine |
| 合并日期 | 2026-10-09 |
| labels | bug, documentation, frontend, mistral, multi-modality, performance, ready, scheduler, streaming-input, v1 |
| 热度 | 👍 4 · 💬 40 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
Voxtral realtime 三修:profiling 音频改为单流块避免 16GB 启动 OOM;buffer 计数在 max_model_len 前结束流避免 EngineCore 崩溃;断连时 abort 引擎请求释放槽位。

## 问题现象
三个症状:(1) **启动 OOM(#38233)**:profiling 会跑一条 `max_model_len` token 的音频,默认 `max_model_len`=131072 时在 16 GB 显卡上启动即 OOM。(2) **引擎崩溃**:streaming 会话到达 `max_model_len` 上限时直接杀死 EngineCore(`illegal memory access`),连带所有其他会话一起挂掉。(3) **WebSocket 断连泄漏**:仅取消 generation task 不会 abort 引擎请求,槽位一直被占——停在 VAD-waiting 的请求持续占槽并向已死的 websocket 生成,永不释放、饿死其他流。

## 根因
正文披露:① profiling 输入按整条 `max_model_len` 音频构造,远超实际流式更新尺寸;② buffer 不计数提交 token,超限的流把越界输入送进引擎;③ 清理路径只 `task.cancel()`,而取消 asyncio task 不会 abort 底层引擎请求。

## 修复方案
1. 新增 `VoxtralRealtimeProcessingInfo`(继承 `VoxtralProcessingInfo`):每次流式更新只带 1 个音频(`{"audio": 1}`),profiling 音频取最大流式块(前缀,39 token,由 `get_max_audio_array_len` 按前缀 token 数换算)。item 尺寸仍保持 `max_model_len`——它同时决定 encoder cache 大小:暂停会话会保留最后一块直到恢复,cache 只按一块 sizing 会让并发会话死锁(0/12 @ `--max-num-batched-tokens 256`)。
2. `VoxtralRealtimeBuffer.__init__` 接收 `max_model_len`,`get_input_stream()` 累计 `n_tokens`,达到上限即 `return` 结束流,会话正常收尾(客户端收到 `transcription.done`)。关键 diff 摘录:
```python
+            n_tokens += num_tokens
+            if self._max_model_len is not None and n_tokens >= self._max_model_len:
+                logger.warning(
+                    "Stopping realtime audio stream: reached max_model_len (%d).",
+                    self._max_model_len,
+                )
+                return
```
3. `connection.py` cleanup 记录并 `await self.serving.engine_client.abort(self.request_id)`,再 cancel 并 await generation task;`send_error` 在 socket 已关时用 `contextlib.suppress` 容错。
2026-10-05 更新:按 review 重做,不再动 core——scheduler/runner/output_processor 改动已撤(#58942 覆盖 ERROR/ABORT,#42502 是通用 scheduler guard)。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响 Voxtral realtime 流式转写入口。
- 回归面:`vllm/model_executor/models/voxtral_realtime.py`(multimodal processor 注册的 info 类、buffer、`buffer_realtime_audio` 传参)与 `vllm/entrypoints/speech_to_text/realtime/connection.py`(清理与错误发送路径);非 realtime Voxtral 路径不受影响。测试(RTX 4090 Laptop 16 GB、sliding window 256、PIECEWISE):`--max-model-len 131072` 主线 profiling OOM、本 PR 可启动;12 并发会话 12/12;`--max-model-len 512` 70s 流到顶正常停止;`pytest -k max_model_len` 2 passed(无 buffer cap 时 `test_voxtral_realtime_generator_max_model_len` 报 `EngineDeadError`)。

## 关联
- #38233(boot OOM issue)
- #45833(unbounded-duration 部分,stacked)
- #58942(ERROR/ABORT 覆盖)
- #42502(通用 scheduler guard)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
