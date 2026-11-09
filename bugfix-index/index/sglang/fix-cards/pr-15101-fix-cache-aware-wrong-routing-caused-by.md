# PR #15101: Fix cache aware wrong routing caused by incorrect load tracking

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15101](https://github.com/sgl-project/sglang/pull/15101) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-12-15 |
| labels | run-ci, model-gateway |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
sgl-router cache_aware 策略下负载计数在多条路径漏减导致路由错误;用 RAII guard `WorkerLoadGuardV2` 统一管理增减。

## 问题现象
正文未描述具体现象(Motivation/Modifications 均为空模板)。按 PR 标题,cache aware 路由因负载跟踪(load tracking)不正确而做出错误路由。

## 根因
正文未披露根因。

按标题与 diff 推断(由 diff 推断):cache_aware 策略会在派发请求时对 worker `increment_load`,而旧代码用布尔标志 `load_incremented` 手动在多条路径做 `decrement_load`:请求出错时、非流式成功时、流式收到 `data: [DONE]` 时、以及流式循环结束后的兜底;`send_typed_request` 的重试路径注释也承认"it only decrements on success or non-retryable failures",可重试失败时靠调用点补救。任何一条路径漏减(如流式未出现 [DONE] 结束标记、错误分支遗漏)都会让 worker 负载虚高,cache_aware 依据负载做出的路由随之出错。

## 修复方案
1. `sgl-model-gateway/src/core/worker.rs`:新增 `WorkerLoadGuardV2`,RAII 封装——构造时 `increment_load()`,`Drop` 时 `decrement_load()`;同时移除 `record_outcome` 中重复更新 CB gauge 的代码。
2. `sgl-model-gateway/src/routers/http/router.rs`:删除所有手动 decrement 分支(`load_incremented` 布尔、`worker_for_cleanup` 克隆、错误分支减载、流式 [DONE] 检测减载等),统一改为 `Option<WorkerLoadGuardV2>`:非流式请求在返回前 `drop(load_guard)`,流式请求由 spawn 的转发任务在流结束时 drop。

```diff
-        let load_incremented = if policy.name() == "cache_aware" {
-            worker.increment_load();
-            true
-        } else {
-            false
-        };
+        let load_guard =
+            (policy.name() == "cache_aware").then(|| WorkerLoadGuardV2::new(worker.clone()));
```

3. `circuit_breaker.rs`:熔断器创建时即以 `RouterMetrics::set_cb_state` 写入初始 Closed 状态。错误路径若在构造 guard 后提前 return,guard 同样按 RAII 自动减载。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅涉及 sgl-model-gateway(Rust 路由网关):cache_aware 的负载计数时序变为 guard 生命周期决定;流式请求的负载现在保持到流真正结束,语义与之前"手动多处减载"不同;`record_outcome` 不再刷新 CB gauge,指标更新点移到熔断器内部。非 cache_aware 策略不增载,行为不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
