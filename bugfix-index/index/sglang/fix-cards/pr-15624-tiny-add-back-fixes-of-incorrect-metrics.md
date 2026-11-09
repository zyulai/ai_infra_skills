# PR #15624: Tiny add back fixes of incorrect metrics after worker removal

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15624](https://github.com/sgl-project/sglang/pull/15624) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-12-24 |
| labels | run-ci, model-gateway |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
worker 被移除后其 metrics gauge 残留旧值;新增 `remove_worker_metrics` 在注销 worker 时清零/复位相关指标。

## 问题现象
正文未描述具体现象(Motivation/Modifications 为空模板)。按 PR 标题,worker 被移除后 metrics 值不正确(残留被移除 worker 的旧指标)。

## 根因
正文未披露根因。

机制(由 diff 推断):`WorkerRegistry` 移除 worker 时只调用 `worker.set_healthy(false)` 并注销,没有清理该 worker URL 维度的 Prometheus gauge(`smg_worker_cb_consecutive_failures`、`smg_worker_requests_active` 等),这些 gauge 是按 worker label 常驻的,worker 移除后仍会以最后值继续暴露,导致监控数据不正确。

## 修复方案
1. `sgl-model-gateway/src/observability/metrics.rs`:新增 `Metrics::remove_worker_metrics(worker_url)`,将 `smg_worker_cb_consecutive_failures`、`smg_worker_cb_consecutive_successes`、`smg_worker_requests_active` 置 0;`smg_worker_cb_state` 与 `smg_worker_health` 置 -1(注释说明这两个指标取 0 有特殊合法含义,故临时置 -1,待 metrics-rs 上游支持彻底删除后改为移除,引用 [metrics#653](https://github.com/metrics-rs/metrics/issues/653))。
2. `sgl-model-gateway/src/core/worker_registry.rs`:在移除 worker 的分支中,`worker.set_healthy(false)` 之后调用 `Metrics::remove_worker_metrics(worker.url())`。

```diff
             worker.set_healthy(false);
+            Metrics::remove_worker_metrics(worker.url());
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅涉及 sgl-model-gateway 的指标清理路径:worker 注销时五个 gauge 被复位;若监控面板对 `smg_worker_cb_state`/`smg_worker_health` 出现的 -1 值没有预期,需要相应调整告警规则。路由与请求处理逻辑不受影响。

## 关联
正文未列出(diff 注释中引用了上游 metrics-rs issue https://github.com/metrics-rs/metrics/issues/653)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
