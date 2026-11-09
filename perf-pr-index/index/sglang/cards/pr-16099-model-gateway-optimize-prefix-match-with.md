# PR #16099: [model-gateway]: optimize prefix_match with zero-copy tenant and deferred char count

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @slin1237 |
| 合并 | 2025-12-29 |
| 分类 | Other (tier: major) |
| Label | run-ci, model-gateway |
| Diff | pack 未提供 diffstat（diff 内容可用：2 个 Rust 文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/16099 |

## 一句话结论

model-gateway 的 radix tree prefix_match 去掉字符串分配与开头 O(n) 计数，MATCH 延迟 ~64µs -> ~22µs（3x）。

## 动机

cache-aware 负载均衡策略对每个请求做前缀匹配，旧 `prefix_match_with_counts` 一进来就对整段输入做 O(n) `chars().count()`，匹配后还要 `take_chars` 构造 `matched_text`、把 tenant `to_string()` 克隆为 `String`；tenant 选取又对 `tenant_last_access_time` DashMap 逐 shard 迭代。worker 数越多 MATCH 延迟越差（500 workers 时 ~64µs），是网关热路径上的分配与锁竞争开销。

## 优化点

三处机制改动：1) `PrefixMatchResult.tenant` 从 `String` 改为 `TenantId = Arc<str>`，匹配结果零拷贝返回（legacy `prefix_match` API 需要字符串时再单独 `to_string()`）；2) 删除 `matched_text` 字段，匹配过程不再 `take_chars` 构造前缀字符串；3) `input_char_count` 延迟到匹配结束后按 `matched_chars + remaining.chars().count()` 计算，消除开头对全输入的 O(n) 扫描。另外每个节点新增 `last_tenant` 缓存（`parking_lot::RwLock<Option<TenantId>>`），匹配命中时 O(1) 取缓存 tenant 并校验其在 DashMap 中仍存在，失效才回退 O(shards) 迭代并回填缓存。

## 关键改动

```text
sgl-model-gateway/src/policies/tree.rs: PrefixMatchResult 改 TenantId/去 matched_text/延迟 input_char_count；Node 增加 last_tenant 缓存
sgl-model-gateway/src/policies/cache_aware.rs: 调用点适配 result.tenant.to_string()
```

关键机制（摘录）：
```rust
// Compute input char count from matched + remaining (deferred from start)
let input_char_count = matched_chars + remaining.chars().count();
```

## 性能声明

Benchmark 500 workers：Before MATCH ~64µs（随 worker 数增加而恶化）；After MATCH ~22µs（flat scaling，不随 worker 数变化），即 3x improvement。硬件、模型、请求分布等其他测试条件 PR 未注明。无准确度声明（网关路由逻辑，不影响模型输出）。

## 验证与回归风险

PR 正文未列新增测试（Checklist 均未勾选）。风险点：`last_tenant` 缓存选取的是 DashMap 迭代的"第一个" tenant，与旧行为一致但语义上是不确定的任意者，缓存只是冻结了这一选择（含 `contains_key` 存在性校验）；无匹配节点时 tenant 返回新分配的 `Arc::from("empty")`；legacy `prefix_match` 路径仍有分配开销但仅兼容调用方使用；cache_aware 策略选择行为不变。

## 关联

PR 正文未提及关联 PR；同区域为 model-gateway 的 cache-aware 路由其他优化，参见索引同分类条目。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
