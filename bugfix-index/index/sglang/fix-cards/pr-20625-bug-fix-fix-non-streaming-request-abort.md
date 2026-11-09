# PR #20625: [Bug Fix] Fix non-streaming request abort failure when --enable-metrics is enabled

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20625](https://github.com/sgl-project/sglang/pull/20625) |
| 作者 | fanghao566 |
| 合并日期 | 2026-03-23 |
| labels | model-gateway, run-ci |
| 热度 | 👍 3 · 💬 10 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
--enable-metrics 的 @app.middleware("http") 走 BaseHTTPMiddleware,其 call_next 替换 ASGI receive 使 is_disconnected 恒为 False,非流式请求断连不 abort;用纯 ASGI 适配器透传 receive 修复。

## 问题现象
开启 `--enable-metrics` 时,客户端断连(如 curl 处理中 Ctrl+C)后非流式 `/v1/chat/completions` 请求不会被 abort(日志无 "Abort");不开 `--enable-metrics` 则 abort 正常。复现:1) 带 metrics 启动 server;2) 发非流式请求;3) 请求处理中 Ctrl+C 取消 curl;4) 观察到请求未被 abort。影响所有启用 metrics 部署的非流式请求取消。Fixes #20623。

## 根因
正文披露:`add_prometheus_track_response_middleware()` 使用 `@app.middleware("http")`,底层是 Starlette 的 `BaseHTTPMiddleware`;其 `call_next()` 会把 ASGI `receive` 替换成内部 channel,导致下游 route handler 里 `request.is_disconnected()` 恒返回 False。`tokenizer_manager.py` `_wait_one_response()` 的断连检测依赖 `request.is_disconnected()`,因此失效。这是 Starlette 已知问题(encode/starlette discussions/2094)。

## 修复方案
- 新增 `python/sglang/srt/utils/http_middleware_patch.py`:纯 ASGI 中间件 `_PureASGIDispatch` 提供固定的 `call_next`,把 `receive` 原样透传(不替换),`patch_app_http_middleware(app)` 替换 `app.middleware("http")` 装饰器。
- `utils/common.py`:在注册 `track_http_status_code` 前调用 patch;上游 dispatch 函数本身不改,只替换它收到的 `call_next`,上游后续改动(v0.5.9+ 新计数器等)自动兼容。
- 新增 `test/registered/scheduler/test_abort_with_metrics.py`:验证 receive 透传后 `is_disconnected()` 可用,以及带 metrics 的 abort 与 `/metrics` 计数器。

关键 diff(新文件 `_PureASGIDispatch.call_next` 核心):
```diff
+        async def call_next(_request):
+            nonlocal status_code
+
+            async def send_and_capture(message):
+                nonlocal status_code
+                if message["type"] == "http.response.start":
+                    status_code = message["status"]
+                await send(message)
+
+            await self.app(scope, receive, send_and_capture)
+            return _SentResponse(status_code)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:经 `@app.middleware("http")` 注册的 http 中间件改为纯 ASGI dispatch 执行;响应状态码捕获改由 send 包装实现;非 "http" scope 直接透传;其他 middleware_type 走原实现。

## 关联
- Fixes:#20623
- Starlette 已知问题:https://github.com/encode/starlette/discussions/2094

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
