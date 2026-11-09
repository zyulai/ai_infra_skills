# PR #16259: [model-gateway] Optimize L1 cache insertion with incremental hashing and tokenization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ppraneth |
| 合并 | 2026-01-13 |
| 分类 | Other(major) |
| Label | dependencies, run-ci, model-gateway |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/16259 |

## 一句话结论

model-gateway L1 cache 插入从 O(N²) 重哈希/重分词改为增量 O(N),50 turns 插入快约 20.5x,lookup 略有回退。

## 动机

L1 cache 在处理长多轮对话时有性能瓶颈:原设计对 prompt 中每个 special token 边界都重新哈希并重新 tokenize 整个前缀,插入耗时随对话长度呈 O(N²) 增长,长上下文或高频轮次更新的负载下形成"性能悬崖"。

## 优化点

Rust 侧 `sgl-model-gateway/src/tokenizer/cache/l1.rs` 三项改动:(1) 增量哈希——`blake3::hash(whole prefix)` 换成维护一个 `blake3::Hasher`,每个边界只喂入上一边界到当前边界的 delta 文本,一次线性前向得到全部前缀哈希;(2) 增量分词——只 tokenize 两个"安全边界"(special token 是原子切分点)之间的新文本段,追加到 running tokens 列表,避免每个前缀整体重编码;(3) BOS 处理——special token(如 BOS)只在第一段的 tokenize 中加入,修复了原逻辑在每个缓存前缀条目里重复添加 BOS 的错误。代价在 lookup 侧:为支持增量哈希,`longest_prefix_match` 现在先做一次完整前向把所有边界的哈希预计算出来,再查 cache shard,延迟有所上升但仍为亚毫秒级。

## 关键改动

```text
sgl-model-gateway/src/tokenizer/cache/l1.rs: insert_at_boundaries 与 longest_prefix_match 改为增量哈希/分词
```

```rust
for (i, &boundary_pos) in boundaries.iter().enumerate() {
    let delta_text = &input[last_pos..boundary_pos];
    // 1. Incremental Hash update
    hasher.update(&bytes[last_pos..boundary_pos]);
    let hash_bytes: Blake3Hash = *hasher.clone().finalize().as_bytes();
    // 2. Incremental Tokenization: only first segment adds special tokens
    let segment_encoding = tokenizer.encode(delta_text, (i == 0) && add_special_tokens)?;
    running_tokens.extend_from_slice(segment_encoding.token_ids());
    ...
    last_pos = boundary_pos;
}
```

## 性能声明

Criterion 基准,模拟 ChatML prompt,按对话轮数(scale):

- 插入(消除 O(N²) 瓶颈):2 turns 60.37 µs → 58.13 µs(约 1.04x);10 turns 327.91 µs → 91.95 µs(约 3.5x);50 turns 5.82 ms → 283.43 µs(约 20.5x)。
- Lookup 回退:50 turns 下 `longest_prefix_match` 从 20.25 µs 升到 94.69 µs;PR 认为相对模型推理可忽略。

PR 未注明基准的硬件/机器规格;Accuracy Tests 章节为空,未提供精度或正确性测试结果。

## 验证与回归风险

PR 正文未列单元测试或正确性验证,Checklist 各项均未勾选。风险点:(1) 增量分词依赖"special token 边界处 tokenizer 前缀稳定"这一假设——原注释明确说明"BPE tokenization is not prefix-stable"才选择整前缀重编码,本 PR 以边界为安全切分点绕开,若 tokenizer 对 special token 后接续文本的编码与分段编码拼接不一致,缓存条目的 token 序列可能与整句编码不同;(2) BOS 只加首段的修复改变了既有缓存条目语义,需要与后端 prefix cache 的匹配逻辑一致;(3) lookup 的预计算前向在边界很多时增加固定开销。

## 关联

无(pack 素材未提及其他同分类或前序/后续 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
