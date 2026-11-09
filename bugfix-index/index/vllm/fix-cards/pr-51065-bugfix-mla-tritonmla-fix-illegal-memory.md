# PR #51065: [Bugfix][MLA] TritonMLA: fix illegal memory access on causal multi-token decode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51065](https://github.com/vllm-project/vllm/pull/51065) |
| 作者 | olka-amd |
| 合并日期 | 2026-09-18 |
| labels | bug, kv-cache-manager, ready, rocm |
| 热度 | 👍 3 · 💬 12 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论

TritonMLA 只声明单 token decode,causal 多 token verify 块未展平即进 decode kernel,越界读 block_table/seq_lens 崩溃;改为统一展平并按行表达因果性。

## 问题现象

投机解码 verify 块触发 `Triton Error [CUDA]: an illegal memory access`。触发条件:TritonMLA 的非因果路由(DSpark draft)把 reorder threshold 提到超过 1,而该阈值被用于切分 causal 批次;此时 NVIDIA sm_86 上(triton_mla 是唯一支持 major==8 的 MLA 后端,CUTLASS/FlashInfer 要求 10、FlashAttn 9、FlashMLA 9/10)或无 AITER MLA 的 ROCm(rocm.py 仅回退到 `[TRITON_MLA]`)均无法避开该路径。独立 sm_86 报告见 #51848。

## 根因

正文披露:`TritonMLA` 声明只支持单 token decode,但 spec decode verify 块未展平就到达 `forward_mqa`:`query_len * num_reqs` 行查询对上只有 `num_reqs` 行的 `block_table`/`seq_lens`。`decode_attention_fwd` 每个 query 行启动一个 program 并执行 `tl.load(B_Seqlen + cur_batch)`,因此超过 `num_decodes` 的每一行都是越界读。而 `forward_mqa` 仅在 `if not causal` 下展开表——`TritonMLAMetadataBuilder` 继承 `query_len_support = SINGLE_ONLY`,普通多 token 查询会路由到 prefill,但 DSpark draft 等非因果组在基类 `__init__` 断言 `SINGLE_ONLY ⇒ threshold == 1` 之后改设标志,把阈值抬高,`build()` 便用被抬高的阈值切分了 causal 批次。

## 修复方案

1. `forward_mqa`(triton_mla.py):由 `q.shape[0] // seq_lens.shape[0]` 推导 `query_len`(用 divmod 断言整除),causal 与非 causal 都展开 `block_table`;causal 时按行表达块内因果性——行 t 的 KV 范围为 `seq_len - (query_len - 1 - t)`,padding 行 clamp 到 0;并断言 causal 多 token 不支持 DCP:

```diff
-        if not attn_metadata.causal:
-            query_len = attn_metadata.num_decode_tokens // attn_metadata.num_decodes
-            if query_len > 1:
-                block_table = block_table.repeat_interleave(query_len, dim=0)
+        num_decodes = seq_lens.shape[0]
+        query_len, remainder = divmod(B, num_decodes) if num_decodes else (1, 0)
+        assert remainder == 0, (
+            f"non-uniform decode block: {B} query rows over {num_decodes} requests"
+        )
+        if query_len > 1:
+            block_table = block_table.repeat_interleave(query_len, dim=0)
+            if attn_metadata.causal:
+                ...
+                seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)
```

2. `_cudagraph_support` 改为 `UNIFORM_BATCH`、`query_len_support` 改为 `UNIFORM`,删除按 `non_causal_multi_token_decode` 条件覆写的 `get_cudagraph_support` 与阈值提升;`attn_logits` workspace 按 `max_num_seqs * reorder_batch_threshold` 扩容。
3. 正文说明 `MLAAttentionSpec.merge()` 改为断言各层在 `non_causal_multi_token_decode` 上一致(替代 `any()`),防止非因果 draft 组与 causal target 组合并。
4. 新增 spy 测试 `tests/kernels/attention/test_triton_mla_causal_verify_flatten.py`(无需 GPU)。

## 影响与回归风险

- 影响窗口:正文未给出引入版本;涉及 sm_86 与无 AITER MLA 的 ROCm 两个受影响平台。
- 回归面:改变 TritonMLA 的 cudagraph 支持声明与 reorder 阈值(原先条件提升逻辑被删,causal 路径也走多 token decode),影响所有使用 TritonMLA 的 MLA 模型 decode 路径;普通单 token decode 12 次启动的次数与形状经测试保持不变。附带收益:cudagraph 恢复 FULL 捕获,Kimi-K3+DSpark TP8/MI325 吞吐 13.01 → 61.3 tok/s,all_reduce 自旋从 2.157s 降到 0.061s。DCP 下 causal 多 token 现被显式拒绝。

## 关联

- #51848(独立 sm_86 issue,2026-08-11 关闭,由本 PR 取代)
- #47310(更早提出 query_len_support 部分,已 stale)
- #51171(仅修非因果 draft 组的 cudagraph 声明,与本 PR 兼容)
- #51705(逐字携带本 PR 的 merge() 修复)
- #52218(9 天后的 draft PR,同机制但为子集)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
