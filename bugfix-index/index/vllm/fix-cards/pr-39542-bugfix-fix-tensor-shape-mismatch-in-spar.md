# PR #39542: [Bugfix] Fix tensor shape mismatch in sparse attention with speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39542](https://github.com/vllm-project/vllm/pull/39542) |
| 作者 | santiramos27 |
| 合并日期 | 2026-04-13 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
稀疏注意力索引器两处张量形状不匹配导致 DeepSeek-V3.2+MTP 在 padded decode 批次下 RuntimeError,改用切片后的 seq_lens 与实际 unpacked 行数修复。

## 问题现象
DeepSeek-V3.2-NVFP4 开启 MTP 投机解码、且 decode 批次带 padding 时报错:
```
RuntimeError: The expanded size of the tensor (104) must match the existing size (98) at non-singleton dimension 0. Target sizes: [104, 2048]. Tensor sizes: [98, 2048] at sparse_indexer.py:168
```
触发需同时满足三个条件:投机解码激活(MTP/EAGLE,next_n > 1);`requires_padding=True`(短于 decode 阈值的 chunked prefill 混入 decode 批次);padding 后批次大小与原始 decode token 数不一致。无投机解码(next_n=1)或无 padding 时尺寸恰好对齐,bug 处于潜伏状态。

## 根因
正文披露:此前一次重构为 `fp8_paged_mqa_logits` 正确地把 `decode_metadata.seq_lens` 切到 `batch_size`,但遗漏了两个下游消费者:
1. `persistent_topk` 仍收到完整(未切片)的 `decode_metadata.seq_lens`,而 logits 与 topk_indices 按 `batch_size * next_n` 尺寸分配;kernel 用未切片 seq_lens 决定输出行数,产出行数少于 topk_indices 期望。
2. `unpack_seq_triton` 解包后按 `topk_indices_buffer[:num_decode_tokens]` 回写,但 `num_decode_tokens` 是 packing 前的总 token 数,而 unpack 返回 `sum(decode_lens)` 行——`requires_padding=True` 时后者更小,导致形状不匹配。

## 修复方案
改 `vllm/model_executor/layers/sparse_attn_indexer.py` 两处:
1. `persistent_topk` 调用改传已切片的 `seq_lens`(替换 `decode_metadata.seq_lens`);
2. 回写改用实际解包行数:
```python
-            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
+            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                 topk_indices
             )
```

## 影响与回归风险
- 回归面:仅 `sparse_attn_indexer` 中 DeepSeek 稀疏注意力索引路径的两处调用/回写;影响启用 sparse attention + 投机解码 + padding decode 的模型(如 DeepSeek-V3.2)。不投机、不 padding 时行为不变(尺寸本就一致)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
