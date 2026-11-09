# PR #38174: [Feature] Universal speculative decoding for heterogeneous vocabularies (TLI)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wan-danfeng |
| 合并 | 2026-07-02 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, speculative-decoding, ready, v1, verified |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/38174 |

## 一句话结论

实现 TLI(Token-Level Intersection)speculative decoding:target 与 draft 词表不同(但有交集)也可 draft,Qwen2.5-1.5B+0.5B 平均接受长度 2.83/3。

## 动机

vLLM 此前要求 draft 与 target 模型共享词表(`verify_equal_vocab_size`),无法跨模型家族/不同 tokenizer 组合 draft 模型。TLI 算法(ICML 2025 oral,arXiv:2502.05202)证明:只要把 draft 约束到两词表的交集,rejection sampling 保持无损,词表不必相同。

## 优化点

启动时构建归一化词表交集:运行时探测 tokenizer 空格前缀(BPE `Ġ` U+0120 / SentencePiece `▁` U+2581,探测失败回退两前缀),归一化 token 字符串后取交集,生成 draft→target 与 target→draft ID 映射张量及交集 mask。draft 生成时 logits 对非交集 token 置 -inf 后采样,再把 draft token ID 映射回 target 空间,rejection sampling 逻辑不变(可证无损);`propose` 与 first-pass 输入经 `map_target_to_draft_ids` 映射进 draft 词表空间。speculative_config 新增 `use_heterogeneous_vocab`:仅支持 `method=draft_model` 且仅 greedy draft sampling(probabilistic 暂不支持,assert 保护并留 TODO);越界 token 回退 unk_token_id(无则 eos_token_id,两者皆无则报错);draft 模型使用自身 tokenizer。

## 关键改动

```text
vllm/v1/spec_decode/vocab_mapping.py(新): VocabMapping 交集构建/双向映射/constrain_draft_logits
vllm/config/speculative.py: use_heterogeneous_vocab 校验(方法与采样限制)、draft tokenizer 独立
vllm/v1/spec_decode/llm_base_proposer.py: _greedy_sample/_sample_draft_tokens/propose/set_inputs_first_pass 接入映射
vllm/v1/spec_decode/draft_model.py: 构建 VocabMapping;非异构时保留原 vocab size 校验
tests/v1/spec_decode/test_vocab_mapping.py(新): 真实 tokenizer 空格前缀探测与回退
```

## 性能声明

Qwen2.5-1.5B + Qwen2.5-0.5B,A800 80GB:词表交集 151665/151936 = 99.8%;mean acceptance length 2.83/3;逐位 acceptance 77.5% / 60.6% / 45.1%;平均 draft acceptance 61%。回归:无 spec decode(baseline)、`ngram`、`draft_model` 均不受影响。PR 未注明吞吐对比数据。

## 验证与回归风险

tests/v1/spec_decode/test_vocab_mapping.py(SmolLM2/TinyLlama/Qwen2.5 真实 tokenizer 前缀探测、损坏/空编码回退);文档与 examples 更新(use_heterogeneous_vocab 示例)。风险:交集过小(<100 token)仅告警不阻断,接受率可能崩塌;异构词表 + probabilistic draft sampling 未实现;映射依赖 unk/eos 回退的正确性;-1 填充映射的越界 token 语义。

## 关联

同分类 Speculative Decoding:#43091(draft_model 的 MRV2 支持)、#20401、#50272、#45635;算法来源 Timor et al. arXiv:2502.05202(HF transformers PR #35029 有参考实现)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
