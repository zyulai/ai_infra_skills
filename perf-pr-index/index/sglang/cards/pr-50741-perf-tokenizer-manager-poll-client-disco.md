# PR #50741: [Perf] tokenizer_manager: Poll client disconnects once per wait timeout, not per streamed chunk

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @nvpohanh |
| 合并 | 2026-11-17 |
| 分类 | Frontend / Serving(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/50741 |

## 一句话结论

修复 #46586 引入的每个流式 chunk 前都轮询 `is_disconnected()` 导致的高并发流式吞吐下降 10–25%:改为每个 `SGLANG_REQUEST_STATE_WAIT_TIMEOUT`(默认 4 s)最多轮询一次,GB200 上 gpt-oss-120b TP4 8k-in/1k-out c256 吞吐从 main 的 6,992 恢复到 8,894 tok/s(+27.2%),断流中止行为保留。

## 动机

#46586(c81230bffe)让 `TokenizerManager._stream_one_response` 在每个流式 chunk 前调用 `await request.is_disconnected()`;此后高并发流式吞吐跨模型跨平台掉了 10–25%(GB200 8k/1k c256:-22% 于该提交、-23% 于当前 main,二分定位到它)。根因:Starlette 的 `is_disconnected()` 是在已取消的 anyio CancelScope 里 `receive()`——每次调用挂起流任务、等一整个 event-loop pass 让 `call_soon` 投递 `Task.cancel()`、再沿调用栈展开 CancelledError。默认 `stream_interval=1` 时每个 token 一次,全部压在同时负责 HTTP、tokenize 8k prompt、扇出输出的单个 TokenizerManager 事件循环上。微基准(FastAPI/uvicorn,256 条 SSE 流,单 GB200 Grace 核):每 chunk 服务器 CPU 14→42 us,可持续 chunk 速率约 68k/s→24k/s。GPU 调度器不受影响(~28.6k tok/s 两构建相同),损失在调度器与客户端之间,客户端 ITL 升 35%。

## 优化点

- `_stream_one_response` 的逐输出断连轮询改为每 `SGLANG_REQUEST_STATE_WAIT_TIMEOUT` 最多一次(`next_disconnect_poll` 时间戳门控;首个输出仍轮询,`next_disconnect_poll = 0.0`,捕获排队/prefill 期失联;该超时已是另一条轮询路径的节奏,不引入新常量)。
- 不完全回滚的理由:ASGI spec ≥2.3 的 uvicorn/Granian 与 gRPC 已有事件驱动断流检测(实测所有构建在三个 stream 端点上 0.14 s 内释放断流),但保留限频轮询可为 ASGI 服务器/代理从不报断连的部署兜底,剩余轮询开销至多约 1.3%,在噪声内。
- 评估过并否决的方案:纯 revert(失去兜底)、把探测本身做便宜(ASGI 没有非阻塞探测;并发 receive watcher 会与 Starlette 的 `listen_for_disconnect` 竞争)、新增调参旋钮(不需要)。

## 关键改动

```python
+        next_disconnect_poll = 0.0
         while True:
...
+            now = time.monotonic()
             if (
                 request is not None
                 and not obj.background
-                and await request.is_disconnected()
+                and now >= next_disconnect_poll
             ):
+                next_disconnect_poll = now + _REQUEST_STATE_WAIT_TIMEOUT
+                if await request.is_disconnected():
                     self.abort_request(obj.rid)
                     raise ValueError(...)
```

新增两个单测:`test_stream_polls_disconnect_once_per_wait_timeout`(在一个 wait 超时内 8 个输出只 await 一次,在 main 上失败)与 `test_stream_polls_again_once_wait_timeout_elapses`(超时后再轮询仍会 type 3 中止)。

## 性能声明

(全部照抄 PR 正文,未独立核实;1×GB200 NVL tray,gpt-oss-120b TP4,流式 /v1/completions 8000 in/1000 out,并发 256,1280 请求)

- 输出吞吐 tok/s:父提交 6ec5451954 均值 9088;c81230bffe 7116(−21.7%);main(f76ac26543)6992(−23.1%);c81230bffe + 本 PR 8818(−3.0%);main + 本 PR 8894(−2.1%,即 +27.2% vs main)。修复恢复缺口的 86–91%。
- 控制实验:仅首输出轮询(timeout=3600)8864;完全去掉输出路径轮询 8937——剩余轮询最多耗约 1.3% 吞吐,在噪声内。
- 配对 A/B(并发 256,4 runs/build):base 6654 → PR 8362(+25.7%),ITL mean 32.75→25.20 ms;并发 1 两者持平。
- 断连行为:三种 stream 端点所有构建(含父提交)0.14 s 内释放;非流式 1.93–1.94 s,不变。

## 验证与回归风险

两个新单测钉住"每超时至多一轮询"与"超时后再轮询仍中止";测试文件 62 passed/47 subtests,`pre-commit` 全过。风险面:若部署的 ASGI 服务器/代理不报 `http.disconnect`,断流释放最坏延迟一个 wait 超时(默认 4 s),这是 #46586 想要的兜底被有意保留;非流式路径不受影响。Base CI 通过。

## 关联

#50722(被修复的 issue)、#46586 / c81230bffe(引入逐 chunk 轮询的提交,其 `_ClosingStreamingResponse` 等事件驱动部分保留)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
