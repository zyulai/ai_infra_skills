# PR #22247: [Anthropic] Fix clock mismatch in received_time causing negative Prometheus metrics

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22247](https://github.com/sgl-project/sglang/pull/22247) |
| 作者 | lixuwei2333 |
| 合并日期 | 2026-04-14 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
Anthropic 入口用 wall clock 记 `received_time`、下游按 monotonic 做差,Prometheus 出负值;改用 `monotonic_time()` 修复。

## 问题现象
在生产环境中发现:Anthropic API 请求在 Grafana 面板上持续产生负的 TTFT 与 e2e latency 值;Prometheus 直方图 `time_to_first_token_seconds` 与 `e2e_request_latency_seconds` 出现大的负值。触发条件:走 Anthropic API 入口的请求(流式与非流式均受影响)。

## 根因
正文披露:Anthropic API 入口(`serving.py`)用 `time.time()`(wall clock,epoch 基准)记录 `received_time`,而下游 `req_time_stats.py` 的 metrics 计算是对 `time.perf_counter()`(monotonic)做差值。两种时钟不匹配,做差产生大负值。OpenAI 入口(`serving_base.py`)已在 PR #17640 中改为使用 `monotonic_time()`,但 Anthropic 入口被遗漏。

## 修复方案
- `python/sglang/srt/entrypoints/anthropic/serving.py`:`_handle_non_streaming` 与 `_handle_streaming` 两个方法中,`received_time = time.time()` 改为 `received_time = monotonic_time()`;新增 `from sglang.srt.observability.req_time_stats import monotonic_time`。
- 正文注明:`received_time_perf = time.perf_counter()` 有意保留——它用于本地计算 `validation_time`,用途不同;而 `received_time` 传给 `req_time_stats` 用于 Prometheus metrics。

```diff
-        received_time = time.time()
+        received_time = monotonic_time()
         received_time_perf = time.perf_counter()
```

## 影响与回归风险
- 影响窗口:OpenAI 入口已在 #17640 修复、Anthropic 入口一直遗漏,故自 Anthropic 入口引入 `received_time` 传给 req_time_stats 起,Prometheus 指标即为负值(基于正文对两入口差异的描述)。
- 回归面:仅 Anthropic 入口两个 handler 的 `received_time` 时钟来源变化,只影响 Prometheus 指标采集,不影响模型输出(正文);`monotonic_time()` 包装 `time.perf_counter()`,性能与 `time.time()` 相当(正文);`received_time_perf` 语义不变。

## 关联
- [#22249](https://github.com/sgl-project/sglang/issues/22249)
- [#17640](https://github.com/sgl-project/sglang/pull/17640)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
