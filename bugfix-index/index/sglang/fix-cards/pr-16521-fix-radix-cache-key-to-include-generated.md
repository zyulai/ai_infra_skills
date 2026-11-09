# PR #16521: Fix radix cache key to include generated tokens in multi-turn (regression)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16521](https://github.com/sgl-project/sglang/pull/16521) |
| 作者 | ycchen-tw |
| 合并日期 | 2026-02-10 |
| labels | high priority, run-ci |
| 热度 | 👍 3 · 💬 7 |
| 分类 | KV Cache / Memory |
| bug 类型 | regression |

## 一句话结论
多轮对话 radix cache 只复用 prompt、不复用上一轮生成 token;cache_finished_req/unfinished 改用含 output 的 token_ids 作插入键。

## 问题现象
多轮使用下 radix cache 只命中原始 prompt。正文复现:Round 1 prompt=100 tokens、生成 50;Round 2 input=200(100 prompt + 50 generated + 50 new tokens),期望 cache hit 150(prompt+generated),实际只 hit 100(仅 prompt)。修复后 Round 2 Cached=150。后果是下一轮对已生成的 continuation token 重复 prefill,显著拖慢多轮性能。示例日志(修复前):`Round 2: Input=200, Cached=100  (only prompt)`。

## 根因
正文披露:`sglang/srt/mem_cache/radix_cache.py::cache_finished_req()` 中 `token_ids` 已正确算出为 `(req.origin_input_ids + req.output_ids)[:kv_committed_len]`,但插入键用的是 `req.fill_ids`;`fill_ids` 在 `init_next_round_input()` 时(decode 前)就固定,不含 `output_ids`,于是 radix cache 只存/匹配 prompt 前缀。此为非 EAGLE 路径的回归。

## 修复方案
`cache_finished_req` 与 `cache_unfinished_req` 的插入键统一改为 `token_ids`,保留 EAGLE 的 bigram-key 行为:

```python
-        keys = convert_to_bigram_key(req.fill_ids) if self.is_eagle else req.fill_ids
+        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
```

## 影响与回归风险
- 影响窗口:正文未说明(标注为回归,即上一版引入)
- 回归面:radix cache 的 KV 复用键选择;正文声明不改动模型 forward/采样,模型输出不变。EAGLE 路径继续走 `convert_to_bigram_key`,行为保持。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
