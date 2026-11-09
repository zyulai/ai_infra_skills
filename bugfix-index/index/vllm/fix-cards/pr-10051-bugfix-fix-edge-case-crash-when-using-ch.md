# PR #10051: [Bugfix] Fix edge-case crash when using chat with the Mistral Tekken Tokenizer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10051](https://github.com/vllm-project/vllm/pull/10051) |
| 作者 | tjohnson31415 |
| 合并日期 | 2024-11-06 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
Tekken tokenizer 把 special token id 传入 tiktoken 越界崩溃;convert_ids_to_tokens 过滤 special token id。

## 问题现象
`vllm serve mistralai/Ministral-8B-Instruct-2410 --tokenizer-mode=mistral` 下,一个普通 chat 请求(user content 为中文「寫一首關於勇敢的水手的詩」)即可让 engine/server 崩溃。调用栈:`decode_sequence_inplace` → `convert_ids_to_tokens` → mistral.py `id_to_byte_piece` → tiktoken `decode_single_token_bytes`,以
```
OverflowError: out of range integral type conversion attempted
```
结束。

## 根因
正文披露:作者 traced 原因是 special token 的 id 被传入 `id_to_byte_piece`:chat template 以 `[/INST]` special token 结尾,prompt 结尾包含在增量 detokenization 的 "prefix" 部分;生成 symbolic language(如中文)的 token 时,代码很可能在首个生成 token 就走「把 token id 转成 bytes」的路径,从而触发错误。

## 修复方案
`vllm/transformers_utils/tokenizers/mistral.py` `convert_ids_to_tokens`:对 Tekkenizer 显式过滤 special token id(该路径本就要求 `skip_special_tokens=True`):
```python
if isinstance(self.tokenizer, Tekkenizer):
    # skip special tokens
    ids = [i for i in ids if i > self.tokenizer.num_special_tokens]

tokens = [self.tokenizer.id_to_piece(id) for id in ids]
```
`decode` 与 `convert_ids_to_tokens` 两处 assert 消息同步改为 "skip_special_tokens=False is not supported for Mistral tokenizers."。测试:`test_mistral.py` 把 Ministral-8B-Instruct-2410 加入 `MISTRAL_FORMAT_MODELS`,symbolic 语言提示词新增缅甸语与 `URGENCY🌶️` 用例(后者关联 PR #9625)。

## 影响与回归风险
- 回归面:Mistral Tekken tokenizer 的 `convert_ids_to_tokens`(增量 detokenization 与 decode 路径);SentencePieceTokenizer 不受影响;special token 本就不应出现在输出文本中,过滤不影响正常 token。

## 关联
#9625(测试用例注释中提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
