# PR #3822: [feat] add small vocab table for eagle's draft model[1].

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Zhou-sx |
| 合并 | 2025-03-03 |
| 分类 | Speculative Decoding(tier: major) |
| Label | 无(素材记录 labels: (none)) |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全量保留(5 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/3822 |

## 一句话结论

EAGLE draft 模型改用截断的高频词表(FR-Spec),lm_head 快 7.72×,draft decode 阶段耗时减半,端到端吞吐提升约 7-19%。

## 动机

投机解码中 draft 模型的 decode 层数已大幅减少,计算瓶颈转移到 lm_head(与词表大小成正比)。作者的 profile 显示 draft decode 单次约 1.8ms,其中 lm_head 占 860µs,接近小模型耗时的一半。由于 target 模型验证保证正确性,draft 模型可以安全使用只保留高频 token 的截断词表,降低 lm_head 开销。对应论文 FR-Spec(arXiv:2502.14856)。

## 优化点

新增 `--speculative-token-map` 指向高频 token id 列表(`hot_token_ids.pt`,可直接用 thunlp/LLaMA3-Instruct-8B-FR-Spec 的 freq_32768.pt,或用参考实现从自己数据集统计)。机制:eagle worker 启动时加载 hot_token_id,并通过 `json_model_override_args` 注入 `hot_vocab_size`,使 LlamaEagle 的 lm_head 按 hot_vocab_size 分配;与 target 共享 embedding/lm_head 时把 target head 按 hot_token_id 行索引切片(`head.data = head.data[self.hot_token_id]`);draft_forward 中每步把小词表下的 `topk_index` 通过 `hot_token_id[topk_index]` 映射回全词表 id 再交给验证,保证输出语义不变。NEXTN 路径显式 NotImplementedError。

## 关键改动

```text
python/sglang/srt/speculative/eagle_worker.py: 加载 hot_token_id、切片共享 head、topk_index 映射回全词表
python/sglang/srt/models/llama_eagle.py: config.hot_vocab_size 存在时按小词表建 lm_head
python/sglang/srt/server_args.py: 新增 --speculative-token-map
docs/backend/server_arguments.md, speculative_decoding.ipynb: 用法文档与示例
```

## 性能声明

PR 正文数据(环境:Llama-3.1-8B-Instruct + lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B,1×H20;E2E 另注明 private dataset 用 A100):

- draft decode 中 lm_head:482.01 µs → 62.40 µs,提速 7.72×;draft decode 总耗时 975.01 → 483.43 µs(-50.42%)
- draft extend:3305.40 → 2433.08 µs(-26.30%);target model:14233.15 → 13720.46 µs;整段总计 20962.45 → 18717.86 µs
- 端到端:MTBench 111.51 → 119.27 token/s;private dataset(A100)84.49 → 100.24 token/s

## 验证与回归风险

PR 未附单测/精度评测(checklist 相关项未勾选,正文无 accuracy 数据)。风险:hot_token_id 与 draft model 词表/训练分布强相关,映射错位会静默改变 draft 提议;仅覆盖 LlamaEagle 路径与 EAGLE topk 提议链路,NEXTN 不支持;`hot_vocab_size` 经 `json_model_override_args` 注入 config,与 tie_word_embeddings(直接复用 embed_tokens)路径的交互未在 diff 中处理。

## 关联

论文 FR-Spec(arXiv:2502.14856)、EAGLE(arXiv:2406.16858)。同区域:同分类 Speculative Decoding 的 PR #42297(block verification)、PR #10846(fix eagle radix cache)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
