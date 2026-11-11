# PR #65028: [Perf][Rust Frontend] Serialize SSE chunks before building the event

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @davidteather |
| 合并 | 2026-11-10 |
| 分类 | Frontend / Serving (tier: major) |
| Label | ready, rust |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数为 7 个文件(diff 完整,未见截断) |
| 链接 | https://github.com/vllm-project/vllm/pull/65028 |

## 一句话结论

Rust frontend 的 SSE 流式响应改为先序列化再建 event,每帧序列化快 1.86–2.66×,线上字节完全不变,覆盖所有流式请求。

## 动机

chat completions、completions、generate、Responses 四条路由的流式输出都用 axum 的 `Event::json_data(..)` 直接把 serde_json 产出写进 SSE event。serde_json 写一个 chunk 会产生几十个小片段(典型 230 字节 chat chunk 有 77 次 `write` 调用,其中 59 次只有 1–2 字节),`json_data` 把每个片段直接交给 axum 的 `EventDataWriter::write_buf`,后者对每个片段重复换行/CR 扫描、flag 检查和 `BytesMut` reserve,且 buffer 从 capacity 0 随片段不断增长。SSE 帧构造是 Rust frontend 所有流式请求的公共路径,值得消除这段逐片段开销。

## 优化点

核心机制是把"逐片段写缓冲"换成"一次序列化 + 一次拷贝":先把 payload 用 `serde_json::to_string` 序列化进一个临时 `String`,再走 `Event::data(..)`,使原来的几十次小写入变成一次内联 memcpy,`data()` 只做一次扫描和一次拷贝。该逻辑收敛到 `EventExt::json` 一个 helper(与 `sse_response` 同处 `utils.rs`),doc 注明为何绕开 `json_data`。输出字节不变:compact serde_json 不会产出裸 `\n` 或 `\r`,wire bytes 经断言逐字节一致。代价是每帧多一个临时 `String`,但 event buffer 增长步数减少,每帧堆操作持平或减一(两条路径均为 5–7 次)。此外把 chat completions、completions、generate 三份完全相同的 SSE stream 函数合并成一个泛型 `openai_sse_stream`,Responses 路由单独改用 `.json`。

## 关键改动

```text
rust/src/server/src/utils.rs: 新增 EventExt::json helper 与泛型 openai_sse_stream(合并三路由逻辑)
rust/src/server/src/routes/openai/chat_completions.rs: 删除本地 SSE 函数,改用 openai_sse_stream
rust/src/server/src/routes/openai/completions.rs: 同上
rust/src/server/src/routes/inference/generate.rs: 同上
rust/src/server/src/routes/openai/responses.rs: json_data 换成 .json
rust/src/server/Cargo.toml + rust/Cargo.lock: 引入 easy-ext
```

```rust
#[easy_ext::ext(EventExt)]
impl Event {
    /// Writes the same bytes as [`Event::json_data`], which is about 2.5x slower.
    pub fn json(self, payload: &impl Serialize) -> Event {
        self.data(serde_json::to_string(payload).expect("SSE payload must serialize to JSON"))
    }
}
```

## 性能声明

Criterion 基准(axum 0.8.8,Apple M-series,mimalloc,`critcmp` 单次运行,两变体同二进制,每个 case 先断言经 axum `Sse` 的 wire bytes 一致;区间 ±1% 以内,重复运行复现约 1%):content_token 504.8ns→224.4ns(2.25×)、reasoning_token 503.7ns→223.2ns(2.26×)、first_role 554.3ns→246.6ns(2.25×)、content_escapes 557.8ns→239.3ns(2.33×)、content_120B 568.0ns→305.4ns(1.86×)、finish 616.0ns→254.6ns(2.42×)、tool_args 693.4ns→275.5ns(2.52×)、logprobs_top5_token_ids 2.13µs→799.7ns(2.66×),即 46–62% 提速。端到端:`vllm-rs` + `vllm-mock-engine` + `vllm-bench` 流式 chat(2048 请求、3 轮交替)约 −8% server CPU、+6% req/s;作者注明该运行在共享笔记本高负载下,仅作方向性参考,无 GPU 端到端数字。

## 验证与回归风险

行为保持型改动,无新增测试;`cargo nextest run -p vllm-server` 655 passed、0 skipped,clippy `-D warnings` 与 `cargo fmt --check` 均干净。既有路由测试本就解析流式 `data:` 行;byte identity 是对真实 chunk 类型单独断言验证的。回归风险低:线上字节不变是硬性断言,主要风险面是新增 `easy-ext` 依赖与三路 SSE 逻辑合并后错误事件编码路径(`data: {"error": ...}` + `[DONE]`)的泛型实现是否覆盖原类型特例——两版本语义在 diff 中一致。

## 关联

同区域并发改动:作者检索过同样触碰这些路由文件的 open PR(#63926、#63739、#61275、#64573、#63481、#58783),确认均未编辑这几行,无冲突前序/后续 PR。pack 中无其他同分类性能 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
