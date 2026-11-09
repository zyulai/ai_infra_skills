# PR #45022: [Bugfix][Model] Voxtral realtime: fix boot OOM and engine crash at max_model_len

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @damienlaine |
| 合并 | 2026-10-09 |
| 分类 | Other (tier: major) |
| Label | bug, documentation, performance, frontend, ready, v1, multi-modality, mistral, scheduler, streaming-input |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 3 个文件(模型、realtime 连接入口、测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/45022 |

## 一句话结论

修复 Voxtral realtime 启动 profiling OOM 与流达到 max_model_len 时的引擎崩溃,并清理断连后仍占用槽位的孤儿请求。

## 动机

三个相互关联的问题:一是启动 profiling 用一条 `max_model_len` 长度的音频做 dummy 输入,默认 131072 下在 16 GB 显卡上启动即 OOM(#38233);二是流式会话到达 `max_model_len` 会令 EngineCore 出现 illegal memory access 并崩溃,连带杀死所有并发会话;三是 WebSocket 断开时只取消 asyncio generation task 并不会 abort 引擎请求,处于 VAD-waiting 的请求持续占槽并向已死连接生成输出,饿死其他流。

## 优化点

三处机制修复。第一,新增 `VoxtralRealtimeProcessingInfo`:profiling 音频改为最大的 streaming chunk(prompt 前缀,39 tokens),每个 update 限 `{"audio": 1}`;但 mm item 尺寸仍保留 `max_model_len`,因为它同时决定 encoder cache 大小——暂停的会话会保留最后一个 chunk 的缓存,若 cache 缩到一个 chunk 会导致并发会话死锁(实测 0/12)。第二,`VoxtralRealtimeBuffer` 在 `get_input_stream` 中累计已提交 token 数,达到 `max_model_len` 前主动结束流,使会话正常 finish 而非触发引擎越界。第三,连接 `cleanup` 中显式 `engine_client.abort(request_id)` 并 await 被取消的 generation task,保证槽位释放与生成器清理完成。按 reviewer 意见重构后不再含 scheduler/runner 核心改动。

## 关键改动

```text
vllm/model_executor/models/voxtral_realtime.py: 新增 VoxtralRealtimeProcessingInfo(按 chunk profile、audio 限 1);VoxtralRealtimeBuffer 增加 max_model_len token 上限
vllm/entrypoints/speech_to_text/realtime/connection.py: cleanup 时 abort 引擎请求、await 取消 generation task
tests/models/multimodal/generation/test_voxtral_realtime.py: 新增 buffer 截断与引擎健康两个测试
```

buffer 截断逻辑(节选):

```python
n_tokens += num_tokens
if self._max_model_len is not None and n_tokens >= self._max_model_len:
    logger.warning(
        "Stopping realtime audio stream: reached max_model_len (%d).",
        self._max_model_len,
    )
    return
```

## 性能声明

无标准 benchmark 数字(素材正则未检出性能声明)。PR 正文给出的功能性验证,条件为 RTX 4090 Laptop 16 GB、`--hf-overrides` sliding window 256、PIECEWISE compilation:`--max-model-len 131072` 时 main 分支 profiling OOM 而本 PR 可正常启动;并发 realtime 流的 transcript、吞吐与内存与改动前 runner 版本相同,`--max-num-batched-tokens 256` 下 12 会话 12/12 成功;`--max-model-len 512` 下 70 s websocket 流在上限处正常停止并返回 `transcription.done`,下一会话可用。

## 验证与回归风险

新增测试:`pytest tests/models/multimodal/generation/test_voxtral_realtime.py -k max_model_len`(2 passed);`test_voxtral_realtime_buffer_stops_at_max_model_len` 单测 buffer 截断;`test_voxtral_realtime_generator_max_model_len` 在无 buffer cap 的 main 上以 `EngineDeadError` 失败,作为回归哨兵。测试 fixture 改为支持 per-case engine config 覆盖。回归风险集中在 Voxtral realtime 路径:encoder cache 尺寸刻意保持 `max_model_len` 是保守选择,若后续收紧会影响并发会话调度;`cleanup` 的 abort 逻辑改动只作用于 realtime websocket 入口。

## 关联

stacked 后续 PR #45833(unbounded-duration 部分未含在本 PR);正文引用 #58942(ERROR/ABORT 覆盖)、#42502(通用 scheduler guard)、#38233(boot OOM issue)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
