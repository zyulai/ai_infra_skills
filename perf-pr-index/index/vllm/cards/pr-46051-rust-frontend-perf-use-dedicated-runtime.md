# PR #46051: [Rust Frontend][Perf] Use dedicated runtime for HTTP/request-processing/ZMQ

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @BugenZhao |
| 合并 | 2026-06-23 |
| 分类 | Frontend / Serving（tier: major） |
| Label | ready, rust |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/46051 |

## 一句话结论

Rust 前端拆出 HTTP/Request/ZMQ 专用 runtime，长 prompt 压测吞吐 +30.1%、/health p99.9 -93.3%。

## 动机

在高并发、CPU-heavy 请求预处理与繁忙 engine transport 场景下，Rust frontend 的长 prompt 预处理与 ZMQ transport 会占满 HTTP worker 线程，拖累 `/health` 尾延迟与请求侧吞吐。PR 目标是把主异步工作拆到专用 runtime，避免相互挤占。

## 优化点

把主异步工作分成三个 runtime：HTTP runtime 接收连接、路由轻量端点、写响应；Request runtime 通过 Axum middleware 承载重负载推理/分词端点（请求提取、校验、chat-template 渲染、tokenization、request lowering、engine submission）；ZMQ runtime 驱动 engine-core 的 send/receive、output dispatch、abort 处理与 coordinator 任务。实现上：AppState 持有 request runtime（`VLLM_RS_REQUEST_WORKER_THREADS` 可配），新增 middleware 把 `/v1/chat/completions`、`/v1/completions`、`/tokenize`、`/inference/v1/generate` 等 offload 到它；engine-core-client 新增专用 ZMQ runtime（`VLLM_RS_ZMQ_WORKER_THREADS` 可配），engine sends 与后台 ZMQ loops 跑在其上。middleware 以 `layer_fn` 构造 `RequestRuntimeService` 并按 `OFFLOADED_PATHS` 匹配路径，轻量运维路由仍留在 HTTP runtime；ZMQ runtime 用 `BackgroundShutdownRuntime` 包装，drop 时走 `shutdown_background()` 以免阻塞所在线程，默认线程数 4。

## 关键改动

```text
rust/src/server/src/middleware/offload.rs（新增）: request_runtime_layer 与 OFFLOADED_PATHS，按路径分流重请求
rust/src/server/src/runtime.rs（新增）: build_request_runtime（VLLM_RS_REQUEST_WORKER_THREADS，默认上限 32）
rust/src/engine-core-client/src/runtime.rs（新增）: ZMQ runtime 与 BackgroundShutdownRuntime（默认 4 线程）
rust/src/server/src/state.rs + lib.rs + routes.rs: AppState 持有 runtime 并装配 middleware
rust/src/engine-core-client/src/client.rs + imp.rs: engine sends / 后台 ZMQ loops 迁到 ZMQ runtime
```

```rust
+/// Request paths that are run on the request runtime.
+///
+/// These routes can perform CPU-heavy request preparation, including JSON
+/// extraction, validation, chat-template rendering, tokenization, request
+/// lowering, and engine submission. Lightweight operational routes stay on the
+/// HTTP runtime.
+const OFFLOADED_PATHS: &[&str] = &[
+    // HTTP routes:
+    "/v1/chat/completions",
+    "/v1/completions",
+    "/tokenize",
+    "/detokenize",
+    "/inference/v1/generate",
+];
```

## 性能声明

Setup：Rust frontend + Rust mock engine，模型/tokenizer `Qwen/Qwen3-0.6B`，server+mock 绑 `taskset -c 0-69`、bench `-c 70-119`、health probe `-c 120-139`，baseline `d57888efa`、current `4655131ef`。长 prompt / Request runtime 压力（`ISL=300k`、`OSL=1`、`max_concurrency=1024`、`num_prompts=1024`）：Baseline 176.91 req/s、TTFT p99 5.71s、Health p99.9 127.79ms、Health max 676.46ms → Current 230.17 req/s、4.38s、8.59ms、60.40ms；Delta **Throughput +30.1%**、**TTFT p99 -23.3%**、**Health p99.9 -93.3%**、**Health max -91.1%**。Output-heavy（`ISL=32`、`OSL=512`、`max_concurrency=2048`、`num_prompts=16384`）：1407.99 req/s / 720.89k tok/s → 1447.21 req/s / 740.97k tok/s，Delta **Throughput +2.8%**、**Output throughput +2.8%**、**TTFT p99 -40.7%**。PR 未注明精度。

## 验证与回归风险

Test Plan/Result：`cargo check -p vllm-engine-core-client -p vllm-server` 通过；mock-engine 长 prompt 与 output-heavy 两组基准如上。风险面：runtime 拆分影响 chat/completions/tokenize/generate 及 gRPC 请求路径、ZMQ 后台循环与 coordinator、线程数由 `VLLM_RS_REQUEST_WORKER_THREADS` / `VLLM_RS_ZMQ_WORKER_THREADS` 控制；output-heavy 路径并非全项改善——TPOT p99 2.07→4.88 ms、ITL p99.9 102.27→472.75 ms 变差。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
