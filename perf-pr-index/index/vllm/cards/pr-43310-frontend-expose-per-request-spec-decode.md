# PR #43310: [Frontend] Expose per-request spec decode metrics in generate API

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xhx1022 |
| 合并 | 2026-09-20 |
| 分类 | Speculative Decoding \| tier: major |
| Label | documentation, frontend, ready, v1, rust, scheduler |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/43310 |

## 一句话结论

把 opt-in 的 per-request spec decode metrics 暴露到 token-in/token-out `/inference/v1/generate` API。

## 动机

正文目标是把现有 opt-in 的 per-request speculative decoding metrics，通过与 OpenAI 兼容端点相同的 `metrics.speculative_decoding` 响应形状暴露到 `/inference/v1/generate`。#48915 已提供共享 metrics pipeline 与 OpenAI API 暴露；本 PR 复用该 pipeline 通过独立的 token-in/token-out API 暴露，无需另一个 scheduler accumulator 或第二套 public schema（正文 "Why this is not duplicate work"）。

## 优化点

Full responses 对单序列请求包含 metrics；Streaming 遵循 Chat/Completion：metrics 仅在最终 usage chunk（`choices=[]`）返回一次，且需 `stream_options.include_usage=true`，token chunks 不变；Chat 与 Completion 的 derendering 保留 metrics object；Python 对 `n>1` 抑制 request-level metrics，Rust 拒绝 parallel sampling。需要 `--per-request-spec-decode-metrics`，collection 默认禁用；Python 与 Rust frontend 都覆盖；正文明确 "This does not change model execution or sampling"。新增的 `RequestSpecDecodeMetrics` 在 Rust 侧把原先的 `OpaqueValue` 换成具名字段：`num_spec_tokens`（配置的 `num_speculative_tokens`）、`histogram`（按 accepted draft-token 数索引的 verify-step 计数）、`num_draft_tokens`（总 proposed draft tokens）、`per_step_accepted` 与 `per_step_drafted`（仅 `detailed` 时非空）；diff 注释说明 Python 侧以普通 `@dataclass` 经 msgspec 序列化为 map，acceptance rate 等派生值由 frontend 计算，不在线上传输。

## 关键改动

```text
rust/src/engine-core-client/src/protocol/output.rs: spec_decode_metrics 由 OpaqueValue 改为强类型 RequestSpecDecodeMetrics
rust/src/llm/src/output.rs: Rust frontend 传递 metrics
rust/src/server/src/routes/inference/generate.rs 与 generate/types.rs: generate 路由与响应类型
vllm/entrypoints/scale_out/token_in_token_out/protocol.py 与 serving.py: Python token-in/token-out 响应
vllm/entrypoints/scale_out/derender/serving.py、vllm/renderers/online_derenderer.py: derendering 保留 metrics
```

强类型结构片段（摘自 rust/src/engine-core-client/src/protocol/output.rs 的 diff）：

```diff
+#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
+pub struct RequestSpecDecodeMetrics {
+    /// Configured `num_speculative_tokens`.
+    #[serde(default)]
+    pub num_spec_tokens: u64,
+    /// Verify-step counts indexed by accepted draft-token count.
+    #[serde(default)]
+    pub histogram: Vec<u64>,
```

## 性能声明

PR 正文未给出定量性能数据：正文明确 "This does not change model execution or sampling"，没有 benchmark 数字，也没有模型 / 硬件 / 精度 / workload 等测试条件。

## 验证与回归风险

Python spec-metrics tests 6 passed（full / stream 响应、terminal usage placement、`n>1` suppression）；streaming derender metrics passthrough test passed；Rust `vllm-server` generate tests 62 passed（含 streaming / non-streaming 指标传播与 JSON 序列化）；changed-file pre-commit（mypy、ruff、markdownlint、Rust format）passed；GB200 E2E：Qwen3-0.6B 配 n-gram speculation，full response 与最终 streaming usage chunk 返回相同 metrics，token chunks 无 metrics。回归风险：正文明确不改变 model execution 或 sampling；metrics collection 默认禁用、需 `--per-request-spec-decode-metrics`；Python 对 `n>1` 抑制、Rust 拒绝 parallel sampling；其余路径 PR 未注明。正文的 AI assistance 段说明 Codex 参与实现与测试、Claude 做了多轮只读对抗式评审并仍需人工评审；改动同时覆盖 Python 与 Rust 两套前端，涉及 token-in/token-out 的全量与流式响应、Chat 与 Completion 的 derendering 路径，正文未提供性能对照，也未说明开启指标采集后的额外开销。

## 关联

正文明确线索：#48915 提供共享 metrics pipeline 与 OpenAI API 暴露，本 PR 复用之。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
