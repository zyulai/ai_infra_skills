# PR #32642: [Core] Support `min_tokens` with speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qianlihuang |
| 合并 | 2026-02-26 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/32642 |

## 一句话结论

解除投机解码与 `min_tokens` 采样参数的互斥:在 rejection sampling 中按 draft token 掩码 stop token,使最小生成长度约束在 spec decode 下生效。

## 动机

此前 `_validate_spec_decode` 会在 `min_tokens > 1` 时直接抛 ValueError("not yet supported with speculative decoding"),V1 在开启 spec decode 时也干脆不加载 MinTokensLogitsProcessor 并警告这些参数不生效。对使用 DeepSeek MTP、EAGLE3 等投机解码的用户,`min_tokens`(配合 stop token 掩码,常见于让模型先输出思考/固定前缀)完全不可用,e2e 测试也只能把 min_tokens 注释掉。

## 优化点

机制:在 `MinTokensLogitsProcessor` 上新增 `apply_with_spec_decode(logits, num_draft_tokens)`——把打包后的 logits 行按每请求 draft token 数做 cumsum 定位各请求的行区间,对"剩余生成长度还差 N 个 token"的请求,将其前 N 个 draft 行上的 stop_token_ids/EOS 置 -inf(优先级 min_tokens > stop tokens);rejection sampler 的 `apply_logits_processors` 中对 `non_argmax_invariant` 处理器逐个调用该新接口。配套:`_validate_spec_decode` 移除对 min_tokens 的拒绝(仍保留 min_p、logit_bias);`build_logitsprocs` 在 spec decode 下返回包含 `MinTokensLogitsProcessor` 的集合(此前为空);`LogitsProcessors` 构造改为接受 Iterable。

## 关键改动

```text
vllm/v1/sample/logits_processor/builtin.py: MinTokensLogitsProcessor 新增 apply_with_spec_decode,按 draft 行掩码 stop token
vllm/v1/sample/rejection_sampler.py: apply_logits_processors 接入 MinTokensLogitsProcessor 的 spec-decode 版本
vllm/sampling_params.py: _validate_spec_decode 不再拒绝 min_tokens
vllm/v1/sample/logits_processor/__init__.py: spec decode 下保留 MinTokensLogitsProcessor
```

掩码核心逻辑:

```python
remaining = min_tok - current_len
n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))
if n_mask > 0:
    row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
    all_rows.append(np.repeat(row_indices, n_stop))
    all_toks.append(np.tile(stop_toks, n_mask))
logits.index_put_(logits_slice, self.neg_inf_tensor)
```

## 性能声明

PR 正文无吞吐/延迟等性能数字,属于功能解锁(让 spec decode 用户可以使用 min_tokens,避免因提前 stop 重试整次生成)。Test Result 为功能性验证:DeepSeek-V3.2(MTP,TP8,min_tokens 512/max 640)与 Llama-3.1-8B-Instruct(EAGLE3,min_tokens 256/512)的 curl 输出 completion_tokens 分别为 640、283、690,finish_reason 正确。

## 验证与回归风险

测试:`tests/v1/e2e/test_async_scheduling.py` 重新启用 min_tokens=28(原注释"spec decoding currently doesn't support min_tokens"移除);`tests/v1/logits_processors/test_custom_offline.py` 调整为允许内置 logitsprocs(MinTokens/LogitBias/MinP)存在。风险面:所有经 rejection sampler 的 spec decode 方法(deepseek_mtp、eagle3 等)都会执行新掩码路径,依赖 num_draft_tokens 与打包 logits 行的严格对齐;min_p、logit_bias 与 spec decode 仍互斥,行为未变。

## 关联

同分类相关:pr-13365(n-gram matching 优化)、pr-14930(rejection sampling 优化)、pr-16937(EAGLE3)、pr-24539(efficient padded speculation)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
