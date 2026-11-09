# PR #17575: [model-gateway] Optimize HashRing construction to reduce heap allocations

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ppraneth |
| 合并 | 2026-01-28 |
| 分类 | Other (tier: major) |
| Label | dependencies, model-gateway |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/17575 |

## 一句话结论

model-gateway 一致性哈希环构建改用增量 blake3 哈希,消除虚拟节点键的临时堆分配,1000 worker 场景构建提速约 38%、抖动从 6% 降到 1%。

## 动机

`HashRing::new`(`src/core/worker_registry.rs`)此前用 `format!` 宏为每个虚拟节点生成 `"url#vnode"` 键字符串,每次 registry 更新产生 N × 150 次堆分配(每 worker 150 个虚拟节点);1000 worker 的集群一次更新即产生 **150,000 个临时字符串分配**,造成明显的分配器压力与非确定性延迟尖峰。

## 优化点

把"先拼字符串再哈希"改为"增量喂哈希":对每个 worker 创建一次 `Arc<str>` 并取出字节,虚拟节点循环内直接对 `blake3::Hasher` 依次 `update(url_bytes)`、`update(b"#")`、`update(&(vnode as u64).to_le_bytes())`,`finalize()` 后取前 8 字节转 u64 作为环上位置。整个虚拟节点循环不再有任何临时堆分配,`Arc<str>` 仍在 worker 间共享。

## 关键改动

```text
sgl-model-gateway/src/core/worker_registry.rs: HashRing::new 虚拟节点键
  生成从 format! 字符串拼接改为 blake3 增量 update(url + "#" + vnode 的
  LE u64 字节),位置取 hash 前 8 字节
```

关键片段(diff 摘录):

```rust
let mut hasher = blake3::Hasher::new();
hasher.update(url_bytes);
hasher.update(b"#");
hasher.update(&(vnode as u64).to_le_bytes());
let hash = hasher.finalize();
let pos = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
```

## 性能声明

PR 正文,未模拟条件注明硬件。基准模拟 1,000 workers 测量一次 `HashRing` 构建耗时:

| 指标 | Baseline | Optimized | 改善 |
| --- | --- | --- | --- |
| Mean Execution Time | 34.66 ms | 21.36 ms | ~38% Faster |
| Outliers (Jitter) | 6% | 1% | 83% More Stable |

## 验证与回归风险

- PR 的 Accuracy Tests 章节为空,未附正确性验证;Checklist 各项均未勾选;pack 未记录 CI 状态。
- 需要注意的行为差异(由 diff 可见):虚拟节点键的哈希输入编码变了——旧实现把 vnode 格式化为十进制字符串(`format!("{}#{}", url, vnode)`),新实现写入 u64 小端字节,且哈希后取前 8 字节作位置。节点在环上的具体位置因此改变;PR 正文未讨论该变化对已有路由分布/会话粘性的影响。
- `u64::from_le_bytes(...try_into().unwrap())` 依赖 blake3 输出长度 ≥ 8 字节的常量正确性。
- 影响范围限于 sgl-model-gateway(Rust)组件,不触及 SGLang Python 推理路径。

## 关联

无(pack 素材未提供同区域其他 PR 信息)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
