# PR #43982: [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43982](https://github.com/vllm-project/vllm/pull/43982) |
| 作者 | Dymasik |
| 合并日期 | 2026-06-04 |
| labels | bug, ready, speculative-decoding, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Gemma4+MTP+FlashAttention 部分批次时 per-group block_table 未按真实 batch 切片导致 RuntimeError,修复为切片到 batch_size。

## 问题现象
Gemma4 + MTP + FlashAttention 后端在并发负载、批次部分占用(`num_reqs < num_reqs_padded`,即带 CUDA graph padding 的部分批次)时崩溃,压力测试在 >4 RPS 时触发,报 `RuntimeError: batch_size must be equal to batch_size_k`,堆栈经 `gpu_model_runner.py:propose_draft_token_ids` → `gemma4_mtp.py:forward` → `unified_attention_with_output`。实测:8 并发 4 波共 32 请求,修复前仅 3 个成功、29 个失败,第一波 5/8 返回 500(EngineCore encountered an issue),随后服务不可用(Errno 111 连接失败);修复后 32/32 全部成功。TritonAttention 后端不受影响,修复前后均正常,因为它不做同样的 `batch_size == batch_size_k` 断言。

## 根因
正文披露:`Gemma4Proposer.set_per_group_block_table()` 在 `_prepare_inputs` 期间捕获形状为 `(num_reqs_padded, max_blocks)` 的 block tables;随后 `spec_decode_common_attn_metadata` 经 `.unpadded()` 变为 `num_reqs`,但 proposer 实例上保存的 per-group block tables 从未被切片。当 `build_per_group_and_layer_attn_metadata` 为次要 KV cache group 换入存储的 block table 时,FlashAttention 看到 `page_table.size(0) == num_reqs_padded`,而 `cu_seqlens_q` 隐含 `batch_size == num_reqs`,两者不一致即触发断言。

## 修复方案
`vllm/v1/spec_decode/gemma4.py` 的 `build_per_group_and_layer_attn_metadata`:取 `batch_size = common_attn_metadata.batch_size()`,换入 per-group block table 时切片到 `[:batch_size]`,与 drafter 实际处理的未 padding 批次对齐。关键摘录:

```python
batch_size = common_attn_metadata.batch_size()
...
    # Slice to actual batch size to match cu_seqlens_q dimension.
    # The stored block tables may be padded (num_reqs_padded) from
    # the target forward pass, but the drafter operates on the
    # unpadded batch.
    cm.block_table_tensor = self._per_group_block_tables[gid][:batch_size]
```

## 影响与回归风险
- 回归面:仅 `vllm/v1/spec_decode/gemma4.py` 的 per-group block table 换入路径(Gemma4 MTP draft 的次要 KV cache group 元数据构建);只影响目标前向带 padding 而 drafter 未 padding 的场景,全批次(`num_reqs == num_reqs_padded`)下切片为无操作。其他 proposer 不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
