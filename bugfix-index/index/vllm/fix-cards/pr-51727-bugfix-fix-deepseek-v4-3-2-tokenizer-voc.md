# PR #51727: [Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51727](https://github.com/vllm-project/vllm/pull/51727) |
| 作者 | sfeng33 |
| 合并日期 | 2026-08-10 |
| labels | bug, deepseek, ready |
| 热度 | 👍 3 · 💬 7 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
DeepSeek V4/3.2 tokenizer 的 `__len__` 覆盖重复计入已存在于基础词表的 added token,词表超计导致 guided decoding 崩溃,修复为删除该覆盖。

## 问题现象
DeepSeek V4/3.2 系列模型在启用 guided decoding(structured outputs)时崩溃。正文给出 #50924 的复现命令:`vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --max-num-seqs 64 --reasoning-parser deepseek_v4 --speculative-config.method dspark --speculative-config.num_speculative_tokens 7 --speculative-config.draft_sample_method greedy --structured-outputs-config.backend guidance`(正文未贴出具体崩溃日志)。

## 根因
正文披露:`DeepseekV4Tokenizer.__len__` 超计词汇量。`vllm/tokenizers/deepseek_v4.py:81-82` 返回 `tokenizer.vocab_size + len(get_added_vocab())`;对 DeepSeek-V4-Flash-0731 即 128000 + 1283 = 129283,但 1283 个 added token 中有 3 个 id 低于 128000(本已在基础词表中),真实 token 数是 129280,恰等于 `config.vocab_size`,该覆盖造成双重计数。`vllm/tokenizers/deepseek_v32.py:69` 存在同样的过时写法——它源于「HF fast tokenizer 的 `len()` 不含 added tokens」的年代,而 transformers 5.x 的 TokenizersBackend 下 `len()` 已包含 added tokens,因此该覆盖如今既多余又错误。

## 修复方案
直接删除两个文件中的 `__len__` 覆盖及其专用的局部变量(`added_vocab_size`、`tokenizer_vocab_size`):
- `vllm/tokenizers/deepseek_v32.py`:删除 `_DeepseekV32Tokenizer.__len__`(原返回 `tokenizer_vocab_size + added_vocab_size`,注释称 `</think>` 是 added token)。
- `vllm/tokenizers/deepseek_v4.py`:同样删除 `_DeepseekV4Tokenizer.__len__`。

修复后 `len()` 回落到基类(HF tokenizer)实现,与 `config.vocab_size` 一致。

## 影响与回归风险
- 回归面:仅 DeepSeek V4 与 V3.2 两个自定义 tokenizer 包装类的长度语义。若上游 HF tokenizer 的 `len()` 确实含 added tokens(transformers 5.x 下如此),行为恢复正确;任何显式依赖旧超计数值(129283)的下游逻辑会看到数值变化。

## 关联
Fixes #50924、#51467

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
