# PR #14758: [Perf] Optimize radix tree for cache-aware load balancin

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @slin1237 |
| 合并 | 2025-12-09 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | dependencies, run-ci, model-gateway |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,覆盖 4 个文件 + 1 个新 benchmark,另有 1 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/14758 |

## 一句话结论

优化 sgl-model-gateway 缓存感知路由的 Rust radix tree:插入吞吐提升约 27%-100%,prefix match 延迟降低 16%-31%。

## 动机

cache-aware 负载均衡用 radix tree 记录各 worker(租户)的请求前缀以做 KV cache 亲和路由。原实现对每个节点文本反复做 O(n) 的 `chars().count()` UTF-8 字符计数,租户 ID 用 `String` 反复堆分配,每次 insert/match 都调用 `SystemTime::now()` 产生 syscall,children 的 DashMap 用默认 hasher 对单 char 键开销偏高;此外后台 eviction 线程是无限循环,Drop 时无法停止。

## 优化点

六项改动:(1) `NodeText` 结构缓存字符数,消除每次操作的 O(n) 计数;(2) 租户 ID 用 `Arc<str>` interning,克隆/比较变廉价(`Arc<str>: Borrow<str>` 使 contains_key 免分配);(3) 用 `NodeText::split_at_char` 一次遍历同时算出节点分裂的前后缀,替代 `slice_by_chars`;(4) 全局 `AtomicU64` 时间戳 + 5ms staleness 阈值批量更新,减少 `SystemTime::now()` syscall,同时保证 LRU 精度;(5) children DashMap 换用面向单 char 键的自定义 `CharHasher`(golden ratio 乘法散列);(6) eviction 线程加 `AtomicBool` 优雅关停,Drop 时每 100ms 检查退出标志并 join。

## 关键改动

```text
sgl-model-gateway/src/policies/tree.rs: 核心改动,NodeText/CharHasher/时间戳缓存/Arc<str> 租户 ID。
sgl-model-gateway/src/policies/cache_aware.rs: eviction 线程优雅关停(AtomicBool + join)。
sgl-model-gateway/benches/tree_benchmark.rs: 新增 criterion 基准(insert/match/并发/eviction/UTF-8/多租户)。
sgl-model-gateway/src/policies/mod.rs: tree 模块改为 pub 供 benchmark 使用。
sgl-model-gateway/Cargo.toml: 注册 tree_benchmark。
```

时间戳批量更新核心:

```rust
static CURRENT_TIMESTAMP_MS: AtomicU64 = AtomicU64::new(0);
const TIMESTAMP_STALENESS_MS: u64 = 5;
// cached 值在 5ms 内直接复用,超龄才刷新,~99% syscall 减少
if cached != 0 && now.saturating_sub(cached) < TIMESTAMP_STALENESS_MS {
    return cached as u128;
}
```

## 性能声明

`cargo bench --bench tree_benchmark`(criterion,硬件未注明):insert 吞吐 random_text/10 +27%~+55%,random_text/50 +68%~+96%,random_text/100 +71%~+100%,shared_prefix_100 +81%~+117%;prefix_match cache_hit 时间 -29.6%~-31.5%,cache_miss -27.2%~-27.9%,partial_match -16.0%~-16.6%;encoding ascii_match -25.5%~-28.4%、utf8_match -27.1%~-32.1%;multi_tenant shared_prefix_lookup -12.2%~-13.8%、tenant_specific_match -20.9%~-22.3%;eviction evict_to_half/1000 -3.4%~-9.1%、/10000 -10.6%~-19.5%。并发混合负载 2 线程出现回退 +3.0%~+25.4%(p=0.02),4 线程无显著变化,8 线程改善 -4.2%~-13.0%。

## 验证与回归风险

验证手段为新增的 criterion 基准套件(含并发、UTF-8、多租户、eviction 场景),PR checklist 的单测项未勾选。回归风险:(1) 并发 2 线程混合负载基准显示回退,低并发网关场景需关注;(2) 5ms 时间戳 staleness 会使 LRU eviction 的访问时间有最多 5ms 粒度误差;(3) `TenantId` 从 `String` 改为 `Arc<str>` 且 tree 模块转 pub,是 gateway 内部 API 变更;(4) split 路径改为单次遍历,UTF-8 边界行为有 benchmark 覆盖但无正确性单测。

## 关联

同区域(sgl-model-gateway 路由/缓存优化):PR #14894(worker 选择优化)、PR #16259(L1 cache 插入优化)、PR #17575(HashRing 构建优化)。本批次内无直接前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
