# PR #30632: [main][BugFix] Fixed an accuracy bug of Qwen3-next-MTP when batched inferring

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30632](https://github.com/vllm-project/vllm/pull/30632) |
| 作者 | drslark |
| 合并日期 | 2025-12-14 |
| labels | ready, v1, qwen |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Qwen3-Next-MTP 批量推理时 gdn attention 元数据构建用非稳定 argsort 打乱同掩码 token 顺序,产生大量冗余 token,改为 stable 排序修复。

## 问题现象
Qwen3-Next-MTP 批量推理(batched inferring)时输出大量冗余 token(issue #30631;该 bug 最先在 vllm-ascend 中发现,见 vllm-ascend#4930,并已在 vllm-ascend#4932 中先行修复)。复现:Qwen3-Next-80B-A3B-Instruct、TP=4、`speculative_config={"method":"qwen3_next_mtp","num_speculative_tokens":1}`、temperature=0 跑 4 条 prompt。修复前第 3 条「The capital of France is」输出 `' the of capital isThe the of capital isThe the of capital isThe the of capital of capital'`(大量冗余 token);修复后输出 `' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain'`,其余条目正常。

## 根因
正文未披露根因细节(仅定位到 Qwen3-Next-MTP 批量推理产生冗余 token)。从 diff 描述改动所暗示的机制:`vllm/v1/attention/backends/gdn_attn.py` 的 `build` 中,把 spec token 与非 spec token 分组用的是 `torch.argsort(spec_token_masks)`——非稳定排序不保证相等元素(同为 spec 或同为非 spec 的 token)的相对顺序,批量场景下组内 token 顺序可能被打乱,导致注意力元数据与真实 token 顺序不一致,草稿采样产生冗余 token(由 diff 推断)。

## 修复方案
`vllm/v1/attention/backends/gdn_attn.py` 的 `build`:排序加 `stable=True`,保证分组时保持组内原有相对顺序:

```python
-                index = torch.argsort(spec_token_masks)
+                index = torch.argsort(spec_token_masks, stable=True)
```

后续 `non_spec_token_indx = index[:num_non_spec_tokens]`、`spec_token_indx = index[num_non_spec_tokens:]` 的分组逻辑不变。

## 影响与回归风险
- 回归面:仅 GDN attention 后端元数据构建中的 spec/non-spec token 排序,一次一行的改动;所有依赖该分组路径的模型(Qwen3-Next 等 GDN/hybrid 模型)的 spec token 重排顺序从「任意」变为「稳定」,输出确定性与原意一致;非批量、无 spec token 的场景无影响。

## 关联
FIX #30631;发现于 vllm-ascend#4930;vllm-ascend 侧修复 vllm-ascend#4932

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
