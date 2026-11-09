# PR #52720: [Bugfix] Support MistralCommonBackend tokenizers in structured output

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52720](https://github.com/vllm-project/vllm/pull/52720) |
| 作者 | thanhpt1110 |
| 合并日期 | 2026-08-20 |
| labels | bug, structured-output, ready, mistral |
| 热度 | 👍 2 · 💬 46 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
MistralCommonBackend tokenizer 无法被 grammar backend 消费,致 500 或静默坏词表;在 StructuredOutputManager 统一包装为 MistralTokenizer。

## 问题现象
安装 `mistral-common` 且 transformers 的 `TOKENIZER_MAPPING_NAMES` 把 model_type(`mistral`、`ministral`、`ministral3`、`mistral3`、`mixtral`、`pixtral`、`voxtral`、`voxtral_realtime`)映射到 `MistralCommonBackend`、而 `resolve_tokenizer_args` 未选 `tokenizer_mode="mistral"` 时,structured output 失败,两种形态:
- xgrammar:HTTP 500。`is_mistral_tokenizer()` 为 False,工作正常的 Mistral 路径被跳过,`xgr.TokenizerInfo.from_huggingface()` 抛 `ValueError: Unsupported tokenizer type`;backend 在首个 structured-output 请求时惰性构建,服务正常启动、普通补全不受影响。
- outlines:静默出错。不崩溃,`get_cached_tokenizer` 返回的 `CachedTokenizer` 子类 `convert_tokens_to_string` 回退 `"".join(tokens)`,构建出错误的词表:`mistralai/Mistral-Nemo-Instruct-2407`(`tokenizer_mode="hf"`)的 129044 条 reduced-vocabulary 中 409 条映射到错误 token id、坍缩到 3 个共享 id,grammar 编译在损坏词表上。

## 根因
正文披露:`MistralCommonBackend` 子类化 `PreTrainedTokenizerBase` 而非 `PreTrainedTokenizerFast`,两个 grammar backend 均无法直接消费它。

## 修复方案
`vllm/v1/structured_output/utils.py` 新增 `maybe_wrap_mistral_common_tokenizer()`:是 `MistralCommonBackend` 就包成既有 `MistralTokenizer`(其 `__init__` 本就接受该类型):

```python
def maybe_wrap_mistral_common_tokenizer(tokenizer: TokenizerLike) -> TokenizerLike:
    """The grammar backends cannot consume a `MistralCommonBackend` directly."""
    if not isinstance(tokenizer, MistralCommonBackend):
        return tokenizer
    from vllm.tokenizers.mistral import MistralTokenizer
    return MistralTokenizer(tokenizer)
```

在 `StructuredOutputManager.__init__` 获取 tokenizer 处一次性包装,覆盖所有 backend,词表缓存(按 tokenizer 对象键控)不失效;`backend_xgrammar.py`/`backend_outlines.py` 不改。新增测试 `tests/v1/structured_output/test_mistral_common_tokenizer.py`,9 个用例约 15s。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:v1 structured output 初始化路径;非 Mistral tokenizer 原样返回(测试覆盖)。`guidance` backend 本就自行处理 `MistralCommonBackend`,行为不变(SentencePiece-backed 实例的 `TypeError: Guidance only supports Tekken tokenizers` 与本 PR 无关、前后一致)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
