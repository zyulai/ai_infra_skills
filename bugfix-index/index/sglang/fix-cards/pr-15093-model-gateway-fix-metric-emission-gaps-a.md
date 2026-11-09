# PR #15093: [model-gateway] Fix metric emission gaps and name mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15093](https://github.com/sgl-project/sglang/pull/15093) |
| 作者 | slin1237 |
| 合并日期 | 2025-12-14 |
| labels | run-ci, model-gateway |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
model-gateway 指标 describe/emit 名不匹配致指标悬空,并补齐 active_workers、discovery_updates、tree_size 等指标发射。

## 问题现象
正文以要点形式列出问题,未描述运行时具体症状:`sgl_router_upstream_http_responses_total` 被 describe,但实际 emit 的是 `sgl_router_attempt_http_responses_total`,名字不匹配导致该 describe 的指标从未有对应数据;同时存在多处指标发射缺口(metric emission gaps)——活跃 worker 数、服务发现更新、cache-aware 驱逐线程的 tree size 等均未被发射。

## 根因
正文披露:describe 与 emit 的指标名不一致(describe 的是 `sgl_router_upstream_http_responses_total`,emit 的是 `sgl_router_attempt_http_responses_total`);另有一批指标只注册/描述而从未在任何代码路径发射,形成缺口。

## 修复方案
- `observability/metrics.rs`:describe 名改为与 emit 一致的 `sgl_router_attempt_http_responses_total`;新增 `RouterMetrics::set_active_workers`(设置 `sgl_router_active_workers` gauge);
- `RegisterWorkersStep` 与 `RemoveFromWorkerRegistryStep`:worker 注册/移除后调用 `set_active_workers(worker_registry.len())`;
- `worker_registry.rs`:为 `WorkerRegistry` 新增 `len()` 与 `is_empty()` 方法供指标发射使用;
- `service_discovery.rs`:pod 增加、删除事件分别调用 `record_discovery_update(1, 0)` / `record_discovery_update(0, 1)` 发射 `sgl_router_discovery_updates_total`;
- `cache_aware.rs`:驱逐线程按 worker/tenant 遍历 `tenant_char_count` 调用 `set_tree_size` 发射 `sgl_router_tree_size` gauge。

```rust
pub fn set_active_workers(count: usize) {
    gauge!("sgl_router_active_workers").set(count as f64);
}
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:sgl-model-gateway 的指标发射路径(worker 注册/移除 step、服务发现事件处理、cache-aware 驱逐线程)与 metrics describe 注册表;describe 名由 `...upstream_http_responses_total` 改为 `...attempt_http_responses_total`,依赖旧 describe 名的监控/告警配置需同步。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
