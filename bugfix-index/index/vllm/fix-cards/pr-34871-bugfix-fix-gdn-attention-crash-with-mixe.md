# PR #34871: [Bugfix] Fix GDN attention crash with mixed decode/spec-decode batches

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34871](https://github.com/vllm-project/vllm/pull/34871) |
| 作者 | haosdent |
| 合并日期 | 2026-03-16 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 23 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
chunked prefill + MTP 下 GDN backend 的 decode/spec-decode 互斥断言被混合批次触发崩溃;把非 spec decode 重分类为 prefill。

## 问题现象
Qwen3-Next 上同时启用 chunked prefill 与 speculative decoding (MTP) 时,GDN attention backend 崩溃:

```
AssertionError: num_decodes: 1, num_spec_decodes: 2
```

(Fixes #34845。)

## 根因
正文披露:GDN attention backend 断言 `num_decodes` 与 `num_spec_decodes` 互斥。chunked prefill + spec decode 下,某请求最后一个 prefill chunk 可能 `query_len=1` 且无 spec tokens(它们在上一步 `update_draft_token_ids()` 中因 `is_prefill_chunk` 为 True 已被正确丢弃),该请求被误分类为 regular decode,而同批其他请求是 spec-decode,违反断言。崩溃发生在该 last prefill chunk 步本身,而非后续 decode 步(那一步 `post_step()` 已能提供 draft tokens)。

## 修复方案
`vllm/v1/attention/backends/gdn_attn.py` 的 `build()`:当两者同时大于 0 时,把非 spec decodes 重分类为 prefills(prefill kernel 能正确处理带 initial state 的 1-token 序列,结果一致):

```python
if num_decodes > 0 and num_spec_decodes > 0:
    num_prefills += num_decodes
    num_prefill_tokens += num_decode_tokens
    num_decodes = 0
    num_decode_tokens = 0
```

新增 `tests/v1/attention/test_gdn_metadata_builder.py`,8 个参数化用例覆盖原始 #34845 崩溃场景、纯 spec/regular decode、三类混合、多条重分类、零长 padding、重分类后的 `has_initial_state`。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:GDN metadata 构建的分类逻辑;纯 decode、纯 spec-decode、普通 prefill+spec 等路径由测试覆盖保持不变;被重分类的请求改走 prefill kernel 路径并参与 `has_initial_state` 计算。

## 关联
- Fixes [#34845](https://github.com/vllm-project/vllm/issues/34845)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
