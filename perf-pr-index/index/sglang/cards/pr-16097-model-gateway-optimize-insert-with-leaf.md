# PR #16097: [model-gateway] Optimize INSERT with leaf-only timestamp updates

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @slin1237 |
| 合并 | 2025-12-29 |
| 分类 | Other (tier: major) |
| Label | run-ci, model-gateway |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/16097 |

## 一句话结论

sgl-model-gateway 的 tenant 前缀树 INSERT 只更新叶子时间戳、跳过 O(depth) 中间节点写,INSERT 快 6-7x。

## 动机

此前对 MATCH 已做过同款 O(1) 优化(将 INSERT 与之对齐):前缀树 eviction 只通过 `leaf_of()` 检查叶子节点的时间戳,中间节点的时间戳从不被使用,但 INSERT 路径却沿树逐层对每个中间节点做 `entry`/`insert` 时间戳更新,是每层一次的冗余写放大。

## 优化点

把 INSERT 的时间戳写从"路径上每个节点"收缩为"仅叶子":(1) 根节点从 `insert(epoch)` 改为 `or_insert(0)`——根永不被驱逐,只需保证 tenant 存在;(2) 循环中命中已有节点(split 或 full match)只 `contains_key` 确保 tenant 存在,不存在时插入时间戳 0 并补 `tenant_char_count`,不再写 epoch;(3) 新建叶子节点时才取 `get_epoch()` 并写时间戳;循环因 remaining 耗尽而正常退出时,最后停留的 prev 即叶子,补写一次 epoch;(4) 字符计数惰性化——`chars().count()` 只在建叶子时对剩余 slice 计算,不再循环内累计 `matched_chars`。这消除了每字符深度的 DashMap 写与逐层 epoch 读取,insert 变为与树深无关的均摊开销。

## 关键改动

```text
sgl-model-gateway/src/policies/tree.rs: Tree::insert 重写——root or_insert(0)、中间节点仅 ensure tenant、叶子才写 epoch、惰性 char count
```

```rust
// Loop exited normally (remaining empty) - prev is the leaf node
// Update its timestamp for LRU ordering
let epoch = get_epoch();
prev.tenant_last_access_time.insert(Arc::clone(&tenant_id), epoch);
```

## 性能声明

- INSERT 从 52-64 微秒降到 7-11 微秒(6-7x faster);worker 数增加时扩展曲线平坦("Flat scaling across worker counts")。
- PR 未注明测试条件(模型/硬件/workload/测量的具体方式均未给出)。

## 验证与回归风险

PR 未引入测试;checklist 各项(格式化、单测、文档、benchmark)合并时均未勾选。风险面:语义变化在于中间节点的 `tenant_last_access_time` 对新 tenant 记 0 而非当前 epoch——正确性依赖"eviction 只看叶子时间戳(经 `leaf_of()`)"这一不变量,若未来有代码读取中间节点时间戳会取到 0;split 分支中克隆的 DashMap 已带 tenant 时依赖 `contains_key` 判断避免重复计数;MATCH/eviction 与 INSERT 的不变量耦合,回归需靠 gateway 既有测试覆盖。

## 关联

同区域:sgl-model-gateway `policies/tree.rs` 的前缀树;PR 正文注明此优化与此前对 MATCH 的同款 O(1) 优化配套,pack 未给出该前序 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
