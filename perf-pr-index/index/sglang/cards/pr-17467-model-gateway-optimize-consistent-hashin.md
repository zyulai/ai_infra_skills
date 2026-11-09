# PR #17467: [model-gateway] Optimize consistent hashing hot path to eliminate allocations

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ppraneth |
| 合并 | 2026-01-28 |
| 分类 | Other |
| Label | dependencies, model-gateway |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 consistent_hash.rs、stores.rs,新增 criterion 基准 consistent_hash_bench.rs |
| 链接 | https://github.com/sgl-project/sglang/pull/17467 |

## 一句话结论

消除 model-gateway 一致性哈希热路径的堆分配:get_owners 从每次约 5 次分配降到 1 次,基准显示各集群规模下延迟降低 3.3x-4.2x。

## 动机

`ConsistentHashRing::get_owners` 在限流(rate-limit)检查中被高频调用。原实现每次查询都分配一个新 `HashSet` 并克隆多个 `String`。高吞吐场景下这些瞬态堆分配造成显著内存抖动(memory churn)并抬高尾延迟(P99)。

## 优化点

Rust 侧的分配优化:(1) 零拷贝返回——`get_owners` 的返回类型从 `Vec<String>` 改为 `Vec<&str>`,直接返回环上已有 key 的引用,不再分配新字符串;(2) 去重从 `HashSet` 换成对结果向量的线性扫描——owner 数固定为 3(`NUM_OWNERS`),线性 `contains` 比哈希开销小得多;(3) 用 `Vec::with_capacity(NUM_OWNERS)` 预留容量,保证每次调用恰好一次分配;(4) `RateLimitStore::get_owners` 与 `check_ownership_transfer` 适配新引用 API,仅在必要处(如跨锁作用域)才克隆字符串;失败节点匹配也改为引用比较,不再构造 `String`。`is_owner` 同步改为引用比较。新增 criterion 基准 `benches/consistent_hash_bench.rs` 覆盖 10/100/500 节点规模的 `get_owners` 与 `is_owner`。

## 关键改动

```text
sgl-model-gateway/src/mesh/consistent_hash.rs: get_owners 返回 Vec<&str>,去重改线性扫描
sgl-model-gateway/src/mesh/stores.rs: RateLimitStore 适配引用 API,仅在锁外克隆
sgl-model-gateway/benches/consistent_hash_bench.rs: 新增 criterion 基准
sgl-model-gateway/Cargo.toml: 注册基准
```

核心改动(去重与引用返回):

```rust
pub fn get_owners(&self, key: &str) -> Vec<&str> {
    let key_hash = Self::hash(key);
    let mut owners = Vec::with_capacity(NUM_OWNERS);
    let mut iter = self.ring.range(key_hash..);
    while owners.len() < NUM_OWNERS && owners.len() < total_unique_nodes {
        if let Some((_, node)) = iter.next() {
            if !owners.contains(&node.as_str()) {
                owners.push(node.as_str());
            }
        } else {
            iter = self.ring.range(..); // wrap around
        }
    }
    owners
}
```

## 性能声明

criterion 基准(`benches/consistent_hash_bench.rs`)显示各集群规模下延迟降低 3.3x-4.2x:

| 集群规模 | 修改前 (ns) | 修改后 (ns) | 提升 |
|---|---:|---:|---:|
| 10 Nodes | ~419 | ~100 | 4.17x faster |
| 100 Nodes | ~467 | ~132 | 3.53x faster |
| 500 Nodes | ~480 | ~143 | 3.35x faster |

堆分配从每次调用约 5 次降为 1 次。PR 未提供端到端网关吞吐/延迟数据。

## 验证与回归风险

验证依赖基准与既有单元测试(diff 中可见测试断言同步改为引用比较,如 `owners.contains(&"node1")`)。PR 正文 Accuracy Tests 为空,未新增专门正确性测试。风险点:API 从 owned `Vec<String>` 改为借用 `Vec<&str>`,所有调用方需在锁作用域内处理或显式克隆——`RateLimitStore::get_owners` 对外仍返回 `Vec<String>`(在锁内 map/collect),跨锁使用的正确性依赖调用方自律;线性去重的正确性依赖 NUM_OWNERS(3)远小于环节点数的假设;`check_ownership_transfer` 的失败节点匹配改为双重迭代比较,若 failed_nodes 较大复杂度上升。

## 关联

同区域优化:sgl-model-gateway mesh 体系(consistent hash ring、RateLimitStore、wasm_middleware_latency 基准)。PR 正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
