# PR #14894: [model-gateway] optimize worker selection

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ppraneth |
| 合并 | 2025-12-12 |
| 分类 | Other (tier: major) |
| Label | dependencies, run-ci, model-gateway |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数) |
| 链接 | https://github.com/sgl-project/sglang/pull/14894 |

## 一句话结论

把 sgl-model-gateway 路由热路径中的 worker 统计查询从 O(N) 分配改为 O(1) 查表,5000 worker 规模下单次调用快约 21.2x。

## 动机

`RouterManager.select_router_for_request` 是请求路由的热路径,之前每个请求都调用 `worker_registry.get_all()`。该方法需要为注册表里的每个 worker clone 一个 `Arc` 并分配新的 `Vector` 来容纳它们,复杂度 O(N)。在高负载或 worker 数量很大(如 5000)时,持续的内存分配和原子引用计数带来显著的 allocator 压力与 CPU 开销,限制了 gateway 的最大吞吐。

## 优化点

核心改动是把"遍历并物化全部 worker 列表"换成"利用既有索引直接读计数"。`WorkerRegistry` 新增 `get_worker_distribution()`:通过已有的 `type_workers` `DashMap` 索引以 O(1) 取出 `WorkerType::Regular` 桶的长度,再用 `self.workers.len()` 得到总数,两者相减得到 PD worker 数——全程不迭代、不分配。路由侧 `select_router_for_request` 删除了原来对 `get_all()` 结果做 `filter + count` 的代码,改为一次元组解构,把内存分配彻底移出路由决策路径。选择 router 的打分循环本身不变,只是输入的 `num_regular_workers` / `num_pd_workers` 来源变了。

## 关键改动

```text
sgl-model-gateway/src/core/worker_registry.rs: 新增 get_worker_distribution(),基于 DashMap 索引 O(1) 返回 (regular_count, pd_count)
sgl-model-gateway/src/routers/router_manager.rs: select_router_for_request 改用新方法,删除 get_all() + filter/count
sgl-model-gateway/benches/router_registry_bench.rs: 新增 criterion 基准,对比新旧两种取法
sgl-model-gateway/Cargo.toml: 注册 router_registry_bench 基准
```

关键片段(worker_registry.rs):

```rust
pub fn get_worker_distribution(&self) -> (usize, usize) {
    let regular_count = self
        .type_workers
        .get(&WorkerType::Regular)
        .map(|v| v.len())
        .unwrap_or(0);
    let total_workers = self.workers.len();
    let pd_count = total_workers.saturating_sub(regular_count);
    (regular_count, pd_count)
}
```

## 性能声明

PR 用 criterion 基准对比新旧方法,模拟 5000 worker 的注册表:

- 旧 `get_all()`:约 222.25 µs/次
- 新 `get_worker_distribution()`:约 10.47 µs/次
- 提升约 21.2x

测试条件为模拟注册表 5000 worker(Regular/Decode 各半),未注明硬件与模型;这是路由组件的微基准,非 e2e 数字。

## 验证与回归风险

PR 新增 `benches/router_registry_bench.rs` 作为基准;正文未给出功能/准确性测试结果。回归面集中在 sgl-model-gateway(Rust 网关)的路由决策:新方法的 PD 计数定义为"总 worker 数减 Regular 数",语义与原先"遍历统计非 Regular"一致;若注册表中出现新的 WorkerType 归类方式变化,需保持该不变式。不影响模型输出与 CUDA/内核路径。

## 关联

同属 sgl-model-gateway(model-gateway)路由组件优化,pack 素材中未列出同区域其他 PR 编号,无法确认直接关联,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
