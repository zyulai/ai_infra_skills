# PR #27545: Fix NaN in triton EAGLE spec-v2 draft-extend CUDA graph at topk>1 (wrong qo_indptr stride)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27545](https://github.com/sgl-project/sglang/pull/27545) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-08 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
draft-extend CUDA graph 的 qo_indptr 步长误用 num_steps+1,topk>1 时读错偏移产生 NaN;v2 路径改用 num_draft_tokens。

## 问题现象
triton 后端、EAGLE spec-v2 的 draft-extend CUDA graph 下,topk>1 且 bs>1 时每个请求的 query 从错误偏移读取,产生 NaN;bs==1 与 eager 模式均不受影响。正文未给出具体日志,影响面为 triton 后端 + spec-v2 draft-extend CUDA graph + topk>1 的组合场景。

## 根因
正文披露:draft-extend CUDA-graph metadata 构建 `qo_indptr` 时,每请求步长用的是 `num_steps+1`,但 runner 实际按 `num_draft_tokens` 个 token/请求布局;topk>1 时两者不相等,query 读到错误 offset。

## 修复方案
改 `python/sglang/srt/layers/attention/triton_backend.py` 两处:v2 draft-extend 路径的每请求 token 数与 max_extend_len 都改用 `num_draft_tokens`(max_extend_len 不匹配会导致 extend kernel grid 过小、丢 query block)。

```python
-        num_tokens_per_bs = self.speculative_num_steps + 1
+        # V2 draft-extend fills num_draft_tokens per req (the cuda-graph runner's
+        # token layout); num_steps+1 only equals that when topk == 1.
+        num_tokens_per_bs = (
+            self.num_draft_tokens
+            if forward_mode.is_draft_extend_v2()
+            else self.speculative_num_steps + 1
+        )
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:仅 triton attention 后端的 draft-extend CUDA graph 元数据构建(`_update_draft_extend_buffers`、`_build_cuda_graph_forward_metadata`);非 v2 draft-extend 与 eager 路径保留原 `num_steps+1` 逻辑不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
