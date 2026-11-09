# PR #17387: [model-gateway] Optimize special token search using Aho-Corasick

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ppraneth |
| 合并 | 2026-01-25 |
| 分类 | Other (tier: major) |
| Label | dependencies, run-ci, model-gateway |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 sgl-model-gateway/Cargo.toml、benches/special_token_search.rs(新增)、src/tokenizer/stop.rs |
| 链接 | https://github.com/sgl-project/sglang/pull/17387 |

## 一句话结论

把 sgl-model-gateway 的 stop sequence/special token 搜索从 O(N×M) 嵌套循环换成单遍 Aho-Corasick 自动机,50 个 stop 序列时延迟降约 29 倍。

## 动机

PR 之前 `StopSequenceDecoder` 用朴素嵌套循环(`.find()` + `.starts_with()`)在输出文本中搜索 stop sequences 与 special tokens,复杂度为 O(N×M)(N 为文本长度,M 为 token/序列数量)。随着 stop sequences 数量增加,延迟出现尖刺(spike),对 decode 路径的流式输出不稳定。这是 router/gateway 层每个请求都要走的路径,值得用数据结构优化掉。

## 优化点

核心改动是把多模式串匹配换成 Aho-Corasick 自动机:在 `StopSequenceDecoder` 初始化时把 hidden 与 visible 两类 stop sequences 合并编译成一个 `AhoCorasick` 自动机并预存;运行时对 `jail_buffer` 只做单次 `ac.find()` 匹配,整体复杂度降为 O(N),且与 stop 序列数量无关。由于两类序列共用一个模式索引空间,新增 `visible_boundary_idx` 记录 visible 序列在模式列表中的起始下标,用匹配到的 `mat.pattern()` 下标判断命中是 hidden(截断到匹配起点、输出 `Stopped`)还是 visible(保留到匹配终点、输出 `StoppedWithText`)。无 stop 序列时保持 `None`,行为退化为原逻辑。

## 关键改动

```text
sgl-model-gateway/src/tokenizer/stop.rs: StopSequenceDecoder 预编译 Aho-Corasick 自动机,替换 hidden/visible 两段嵌套 .find() 循环,用 visible_boundary_idx 区分命中类型
sgl-model-gateway/Cargo.toml: 新增 aho-corasick = "1.1.4" 依赖,注册 criterion bench special_token_search
sgl-model-gateway/benches/special_token_search.rs: 新增 naive vs aho 的对比 benchmark(5/50 个 token,文本为对话串 repeat 1000)
```

```rust
let mut patterns: Vec<String> = config.stop_sequences.iter()...collect();
let visible_boundary_idx = patterns.len();
patterns.extend(config.visible_stop_sequences.iter()...);
let aho_corasick = if patterns.is_empty() { None }
    else { Some(AhoCorasick::new(patterns).expect("Failed to build Aho-Corasick automaton")) };
```

## 性能声明

PR 正文给出 criterion benchmark(文本为 "User: Hello! Assistant: ..." 对话串 repeat 1000):

| 实现 | Tokens | 平均延迟 | 吞吐 |
| --- | --- | --- | --- |
| Naive(原实现) | 5 | 30.00 µs | 1.55 GiB/s |
| Aho-Corasick(本 PR) | 5 | 4.74 µs | 9.83 GiB/s |
| Naive(原实现) | 50 | 138.10 µs | 345.28 MiB/s |
| Aho-Corasick(本 PR) | 50 | 4.74 µs | 9.82 GiB/s |

作者结论:50 个 stop sequences 时约 **29x speedup**;复杂度变为 O(N),延迟不再随 stop 序列数量增长。PR 未注明基准运行的硬件条件。

## 验证与回归风险

PR 新增了 criterion bench(naive vs aho 对比),但正文的 Accuracy Tests 一节为空("`.`"),未提供端到端精度/正确性测试结果。风险点:stop/hidden/visible 语义依赖 `visible_boundary_idx` 与模式列表顺序的一致性,若上游对 `StopSequenceConfig` 的两类序列含义有变化需同步;`jail_buffer` 部分匹配("jail")逻辑与单次 `ac.find()` 的交互行为(如跨 chunk 的部分匹配)未在 PR 中展开测试说明。

## 关联

同属 model-gateway/Rust 路径的优化;pack 素材中未提供同分类其他 PR 编号,关联写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
