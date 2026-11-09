# PR #36616: [Bugfix] Fix FlashMLA sparse accuracy with topk_length and zero-init padding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36616](https://github.com/vllm-project/vllm/pull/36616) |
| 作者 | AjAnubolu |
| 合并日期 | 2026-06-16 |
| labels | bug, ready, v1 |
| 热度 | 👍 4 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

FlashMLA sparse 未向 `flash_mla_sparse_fwd` 传 topk_length 做精确掩码,BF16 head padding 又用 new_empty;传入有效计数并零初始化修复。

## 问题现象

正文未描述具体现象,仅说明本 PR 修复 FlashMLA sparse 精度问题(见标题),并 Closes #36524。涉及场景为 FlashMLA sparse 后端的 BF16 KV 与 FP8 KV(分离 prefill/decode)前向路径。

## 根因

正文未披露详细根因分析,只陈述修复内容两点:topk_length 需要传给 `flash_mla_sparse_fwd` 以实现精确的 attention masking;BF16 head padding 改用 new_zeros 而非 new_empty(隐含 new_empty 产生的填充值未初始化,为正文措辞的推断)。由 diff 可见机制:索引转换函数以 `return_valid_counts=True` 返回每个请求的有效 topk 计数,并沿 FP8 decode 的 chunk 切片传递到 `_bf16_flash_mla_kernel` 再传给 `flash_mla_sparse_fwd`(由 diff 推断)。

## 修复方案

修改 `vllm/v1/attention/backends/mla/flashmla_sparse.py`:`_forward_bf16_kv` 与 `_forward_fp8_kv_separate_prefill_decode` 调 `triton_convert_req_index_to_global_index` 时加 `return_valid_counts=True` 接收 `topk_length`;`_fp8_decode` 按 chunk 切片传递;`_bf16_flash_mla_kernel` 新增 `topk_length` 参数并传入 `flash_mla_sparse_fwd`。正文还提到将 BF16 head padding 的 new_empty 改为 new_zeros,该改动未出现在 pack 的 diff 摘录中。

```diff
         topk_indices, topk_length = triton_convert_req_index_to_global_index(
             attn_metadata.req_id_per_token,
             attn_metadata.block_table,
             topk_indices,
             BLOCK_SIZE=attn_metadata.block_size,
             NUM_TOPK_TOKENS=topk_indices.shape[1],
+            return_valid_counts=True,
         )
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:`flashmla_sparse.py` 中 BF16 KV 前向、FP8 分离 prefill/decode 前向与 fp8 decode 分块路径的索引转换与 sparse attention 调用;topk_length 语义变化会改变 `flash_mla_sparse_fwd` 的掩码行为。

## 关联

#36524

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
