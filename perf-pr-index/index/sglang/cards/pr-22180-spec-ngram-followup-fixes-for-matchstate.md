# PR #22180: [Spec][Ngram] Followup fixes for `MatchState` incremental advance

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-04-06 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/22180 |

## 一句话结论
修复 Ngram `MatchState` 增量推进的每 token 堆分配与冗余 resolve,match() 端到端 1.40x(32.4 vs 45.3 us/step)。

## 动机
#21243 的 followup。`advanceMatchState_()` 在每个 token 的循环内新建 `std::vector<NodeRef>`,产生 per-token 堆分配;同时通过 `resolve(state, rootRef())` 访问 root,而这对刚创建的 ref 是一个同义反复检查。这些开销落在 ngram 投机解码的 draft 路径上。

## 优化点
(1) `advanceMatchState_` 把 `next` 缓冲提升到循环外,`reserve(max_trie_depth)` 后每轮 `assign(next_depth, {})` 复用,消除 per-token 堆分配;(2) 直接访问 `root_`(root 永不被驱逐,epoch 已在上方校验),去掉 `resolve(state, rootRef())` 的无效检查;(3) 补充注释固化不变量:`TrieNode::version` 从 1 起使默认构造的 NodeRef(version=0)永不解析到活节点、`reset()` 以 epoch 递增批量失效全部缓存 MatchState、`ngram_worker.py` 中 match state 生命周期(finish/retract 请求的清理语义);(4) 测试从 `spec/utils/` 移到 `unit/spec/` 并从 `register_cuda_ci` 改为 `register_cpu_ci`(ngram corpus 纯 C++,无需 GPU),并新增 `TestNgramCorpusMatchBenchmark` 基准;另在 `rerun-test.yml` 增加 x/N 进度与计时输出。

## 关键改动

```text
python/sglang/jit_kernel/csrc/ngram_corpus/trie.cpp: next 缓冲复用 + 直接访问 root_
test/registered/unit/spec/test_ngram_corpus.py: 新增 TestNgramCorpusMatchBenchmark(增量 vs 全量重建)
.github/workflows/rerun-test.yml: CI 测试循环加进度与计时
```

## 性能声明
CPU benchmark、`max_trie_depth=18`:Incremental 32.4 us/step vs Rebuild 45.3 us/step,Speedup 1.40x。作者说明:`match()` 增量是 O(D) 对比重建 O(D²),但只占 `batch_get` 总延迟的一部分(FFI + `buildRecency` + `fillResult` 占主导),1.40x 的端到端加速与 D=18 下的预期一致。

## 验证与回归风险
`TestNgramCorpusMatchBenchmark` 以 `incremental < rebuild * 1.1` 作为回归断言(容 10% 噪声);既有测试文件整体迁移并改注册到 CPU CI suite(stage-a-test-cpu)。风险:trie.cpp 的不变量依赖(version/epoch/驱逐语义)以注释形式固化,无新增硬断言;root 直访的前提"root 永不被驱逐"若未来变化需同步修改;CI workflow 改动与模型行为无关。

## 关联
前序 #21243(MatchState 增量推进的引入)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
