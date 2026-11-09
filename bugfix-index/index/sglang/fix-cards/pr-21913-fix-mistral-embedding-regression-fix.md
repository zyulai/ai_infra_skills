# PR #21913: fix: mistral embedding regression fix

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21913](https://github.com/sgl-project/sglang/pull/21913) |
| 作者 | dougyster |
| 合并日期 | 2026-04-04 |
| labels | high priority, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
transformers v5 升级后误为 fast tokenizer 恢复 add_eos_token,跳过该恢复使 e5-mistral 嵌入相似度从 ~0.33 回到 1.0。

## 问题现象
使用 `LlamaTokenizerFast` 的嵌入模型(如 `intfloat/e5-mistral-7b-instruct`)在 transformers v5.3.0 升级(#17784)之后余弦相似度从 ~1.0 回归到 ~0.33。作者通过在 v0.5.9(good)与 v0.5.10rc0(bad)之间二分,把首个坏提交定位到 d1e95af28("Upgrade transformers==5.3.0 (#17784)")。

## 根因
正文披露:v5 升级引入 `_fix_v5_add_bos_eos_token()`,用于恢复 v5 在存在 `tokenizer.json` post-processor 时剥离的 `add_bos_token`/`add_eos_token` 标志。该修复对 DeepSeek-V3 等生成模型的 `add_bos_token` 处理正确,但错误地也为 fast tokenizer 恢复了 `add_eos_token`:v4 中 `PreTrainedTokenizerFast` 的 EOS 行为定义在 `tokenizer.json` 的 post-processor 里,Python 属性虽存在但对 fast tokenizer 是 no-op;v5 中 sglang 与 HF vanilla 最终都是 `add_eos_token=False`。仅 sglang 恢复 `True` 导致两边分词器分歧——sglang 追加 EOS,last-token pooling 取到 EOS 嵌入,而 HF 参考取最后内容 token 嵌入,相似度跌到 ~0.33。模型 `tokenizer_config.json` 里的 `"add_eos_token": true` 是因为 Mistral 用 EOS 作生成停止符,并非给 pooling 用的。

## 修复方案
`python/sglang/srt/utils/hf_transformers_utils.py` 的 `_fix_v5_add_bos_eos_token()` 中,当加载的 tokenizer 是 `PreTrainedTokenizerFast` 实例时跳过恢复 `add_eos_token`,与 v5 vanilla 行为一致;`add_bos_token` 的恢复不受影响。

```python
if attr == "add_eos_token" and isinstance(tokenizer, PreTrainedTokenizerFast):
    config_val = _V4_DEFAULTS["add_eos_token"]  # False
```

验证:在 `intfloat/e5-mistral-7b-instruct` 上完整干净重装(sglang + sglang-kernel 卸载重装)后,三组测试的相似度均从 0.334/0.387/0.431 恢复到 1.000/1.000/1.000。

## 影响与回归风险
- 影响窗口:自 d1e95af28(transformers v5.3.0 升级,v0.5.10rc0)起至本修复。
- 回归面:所有经 `_fix_v5_add_bos_eos_token` 加载的 `PreTrainedTokenizerFast`——不再恢复 `add_eos_token`(与 HF v5 一致);slow tokenizer 与 `add_bos_token` 路径不变;若下游曾依赖 sglang 侧为 fast tokenizer 追加 EOS 的行为,需自行适配。

## 关联
- [#17784](https://github.com/sgl-project/sglang/pull/17784)(引入回归的 transformers v5.3.0 升级)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
