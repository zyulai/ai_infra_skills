# PR #59491: [Bugfix][Tokenizer] Fix off-by-one max_token_id from vocab_size

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59491](https://github.com/vllm-project/vllm/pull/59491) |
| 作者 | shijie-lyu |
| 合并日期 | 2026-10-01 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
HF tokenizer 的 `max_token_id` 多算 1,越界 token id 可通过输入校验并触发 CUDA device-side assert;修复为取 `vocab_size - 1`。

## 问题现象
对 embedding 恰好有 vocab_size 行的模型(32000 是 64 行 padding 的倍数,Llama-2 系列拿不到 padding 行),包含 id `vocab_size` 的 prompt 能通过引擎侧 `InputProcessor._validate_model_input` 的校验;TP=1 时 embedding 不对索引做掩码,lookup 失败并抛出 CUDA device-side assert。该问题以 GHSA-q97c-p368-75c4 报告,VMT 评为 moderate 并要求以公开 PR 形式修复。

## 根因
正文披露:`vllm/tokenizers/hf.py` 的 `get_cached_tokenizer` 用 `max(max(get_vocab().values()), tokenizer.vocab_size)` 计算 `max_token_id`,但 `vocab_size` 是计数,对最大 id 为 `vocab_size - 1` 的分词器(Llama-2 系列、GPT-2 等)会返回 `vocab_size`,即越界一。约定上 `max_token_id` 是最大合法 id(含边界):`MistralTokenizer` 取 `vocab_size - 1`,`KimiAudioTokenizer` 返回 `n_vocab - 1`,`SamplingParams` 也以 `max_token_id + 1` 报告词表大小。

## 修复方案
`vllm/tokenizers/hf.py` 的 `get_cached_tokenizer` 中把 vocab_size 分支改为减一:

```diff
-            max_token_id = max(max_token_id, tokenizer.vocab_size)
+            max_token_id = max(max_token_id, tokenizer.vocab_size - 1)
```

`max(get_vocab().values())` 仍是第一来源,所以 added tokens 高于 vocab_size 的分词器(如 Qwen2.5)不受影响。修复前后:TinyLlama-1.1B-Chat-v1.0(vocab 32000)32000→31999;openai-community/gpt2(vocab 50257)50257→50256;Qwen2.5-1.5B-Instruct 保持 151664。并新增测试 `test_cached_tokenizer_max_token_id_is_a_valid_id`(gpt2 上修复前失败:`assert 50257 == 50256`)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:仅改动 `get_cached_tokenizer` 的 vocab_size 分支;依赖旧值 `max_token_id == vocab_size` 的校验路径(`InputProcessor._validate_model_input`、`SamplingParams`)现在会拒绝 id `vocab_size` 的输入;走 `get_vocab()` 主路径的分词器行为不变。

## 关联
- GHSA-q97c-p368-75c4(正文提及的安全通告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
