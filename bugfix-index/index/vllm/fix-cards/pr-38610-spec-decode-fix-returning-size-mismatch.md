# PR #38610: [Spec Decode] fix returning size mismatch on extract hidden states proposer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38610](https://github.com/vllm-project/vllm/pull/38610) |
| 作者 | zzaebok |
| 合并日期 | 2026-04-09 |
| labels | speculative-decoding, ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
extract_hidden_states proposer 在 decode 步直接返回原始 tensor,形状 [N,2] 与期望 [N,1] 不匹配;按 num_speculative_tokens=1 切片只返回目标采样列。

## 问题现象
`extract_hidden_states_proposer` 在 decode 步出现尺寸不匹配(size mismatch):proposer 本应返回 `[batch_size, num_speculative_tokens]` 形状的张量,但当前返回的是原始 tensor(raw tensor)。正文给出具体报错处的形状对比:
```python
self.draft_token_ids_cpu[:num_reqs].copy_(draft_token_ids)
#                   [N, 1]                        [N, 2]  → SHAPE MISMATCH
```
即带 spec token 的 decode 步中 `sampled_token_ids` 形状为 [N,2](target 采样 + spec verification 两列),而拷贝目标只留 [N,1] 的空间,触发形状不匹配。相关背景:作者正在另一个 PR(#38565)里做"返回 mean-pooled vector"的普通响应工作。

## 根因
正文披露:proposer 的 `propose` 返回的是 `sampled_token_ids` 原始张量,没有裁剪到 `[batch_size, num_speculative_tokens]`。由于该 proposer 的 `num_speculative_tokens` 固定为 1,期望形状是 [batch_size, 1];而在带 spec token 的 decode 步里,`sampled_token_ids` 实际形状是 [batch_size, 2](一列 target 采样、一列 spec verification),直接返回导致下游 `copy_` 时 [N,1] 与 [N,2] 尺寸不匹配。

## 修复方案
只改 `vllm/v1/spec_decode/extract_hidden_states.py` 的 `propose` 返回语句:
```python
-        return sampled_token_ids
+        # On decode steps with spec tokens, sampled_token_ids may have
+        # shape [batch_size, 2] (target + spec verification); slice to
+        # return only the target-sampled column.
+        return sampled_token_ids[:, :1]
```
即硬编码切片 `[:, :1]`——正文说明:因为该 proposer 的 `num_speculative_tokens` 被设为 1,所以直接切到第 1 列,只返回 target 采样列。注释同步说明了切片原因。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `vllm/v1/spec_decode/extract_hidden_states.py` 中 extract hidden states proposer 的 decode 路径返回值。切片假定 `num_speculative_tokens == 1`(与该 proposer 的当前设定一致),若未来该配置放宽到 >1 需要改回动态切片;非 decode(prefill 等)路径不受影响。测试:`tests/v1/spec_decode/test_extract_hidden_states.py`,作者报告全部通过。

## 关联
- [PR #38565](https://github.com/vllm-project/vllm/pull/38565)(相关背景:returning mean-pooled vector with the normal response)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
