# PR #59937: [Bugfix][Rust Frontend] Export corrupted request metric

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59937](https://github.com/vllm-project/vllm/pull/59937) |
| 作者 | reidliu41 |
| 合并日期 | 2026-10-12 |
| labels | bug, ready, rust |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
Rust 前端解码了 `num_nans_in_logits` 却在记录请求指标前丢弃,`/metrics` 缺失 `vllm:corrupted_requests_total`;现跨请求全生命周期追踪并导出该计数器。

## 问题现象
`/metrics` 不暴露 `vllm:corrupted_requests_total`,即使 engine-core 已在请求 logits 中检测到 NaN。也就是说,数值损坏(NaN)发生后,运维侧无法通过指标观测到。该指标用于反映可能指示模型、kernel 或硬件问题的数值损坏,Python 前端原本已提供此请求级计数器,Rust 前端缺失。正文未描述具体日志或复现步骤。

## 根因
正文披露:Rust 前端其实已经解码了 `EngineCoreOutput.num_nans_in_logits`,但这个值在记录 request metrics 之前就被丢弃了,没有任何路径把它累计到请求级指标上,因此损坏请求计数从未被导出。

## 修复方案
改 `rust/src/llm/src/request_metrics.rs` 与 `rust/src/metrics/src/request.rs`:
1. `RequestMetricsTracker` 新增 `is_corrupted: bool` 字段(初始 false);每次收到 engine 输出时执行 `self.is_corrupted |= output.num_nans_in_logits > 0`,跨请求生命周期累积损坏状态。
2. 请求结束时,在 `record_request_success` 之后,若 `is_corrupted` 为真则调用 `self.handles.corrupted_requests.inc()`。
3. `RequestMetricHandles` 增加 `corrupted_requests: U64Counter` 句柄,并在 `resolve_request_metric_handles` 中通过 `metrics.corrupted_requests.get_or_create_owned(&labels)` 创建。
4. `rust/src/metrics/src/request.rs` 的 `RequestMetrics` 新增 `corrupted_requests: Family<EngineLabels, U64Counter>`,注册指标 `vllm:corrupted_requests`,描述为 "Corrupted requests, in terms of total number of requests with NaNs in logits."。
5. 测试 `rust/src/llm/tests/generate.rs` 中 `generate_records_request_metrics_in_prometheus_output` 为 `EngineCoreOutput` 填入 `num_nans_in_logits: 2`,并断言 Prometheus 输出包含 `vllm:corrupted_requests_total{...} 1`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 Rust 前端指标路径(`request_metrics.rs`、`metrics/request.rs`)。新增一个计数器及其注册,不改既有指标语义;对每条 engine 输出增加一次布尔或运算,开销可忽略。若下游 dashboard 依赖完整指标列表可能需要同步更新。验证方式为 clippy + nextest 相关测试(含新增断言)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
