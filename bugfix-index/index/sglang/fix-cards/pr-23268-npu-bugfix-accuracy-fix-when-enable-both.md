# PR #23268: 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23268](https://github.com/sgl-project/sglang/pull/23268) |
| 作者 | cen121212 |
| 合并日期 | 2026-04-28 |
| labels | deepseek, npu, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
NPU 上同时开启 NSA CP 与 prefix cache 时推理精度异常;修正 KV 实际序列长度未计入 prefix 长度的问题。

## 问题现象
在 NPU 上同时启用 NSA CP(context parallel)与 prefix cache(prefixcache)时,推理精度异常("when enable both nsa cp and prefixcache, inference accuracy is abnormal")。正文未给出具体数值或日志。

## 根因
正文未披露详细根因,仅说明开启两者组合时精度异常。

机制(由 diff 推断):NSA indexer 的 NPU forward 中,给 attention backend 的 `actual_seq_lengths_kv` 直接使用 `attn_cp_metadata.kv_len_prev_tensor / kv_len_next_tensor`;而开启 prefix cache 时请求带有 prefix(非零 `extend_prefix_lens`),该前缀长度未被计入实际 KV 序列长度,导致 KV 位置偏移、注意力取到错误的 KV 范围,精度异常。此外 `deepseek_v2_attention_mla_npu.py` 中 `fused_split_qk_norm` 快路径在 nsa prefill CP 场景下不应走 `shape[0] < 65535` 的分支(该分支的融合 split/norm 与 CP 拆分不兼容)。

## 修复方案
1. `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的 `forward_npu`:当 `sum(forward_batch.extend_prefix_lens_cpu) > 0` 时,把 prefix 长度加到 kv_len 张量上再写入 metadata:

```diff
-                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
-                    forward_batch.attn_cp_metadata.kv_len_prev_tensor,
-                    forward_batch.attn_cp_metadata.kv_len_next_tensor,
-                )
+                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
+                    total_kv_len_prev_tensor = (
+                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
+                        + forward_batch.extend_prefix_lens.squeeze()
+                    )
+                    total_kv_len_next_tensor = (
+                        forward_batch.attn_cp_metadata.kv_len_next_tensor
+                        + forward_batch.extend_prefix_lens.squeeze()
+                    )
+                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
+                        total_kv_len_prev_tensor,
+                        total_kv_len_next_tensor,
+                    )
+                else:
+                    ...  # 保持原值
```

2. `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` 的 `forward_dsa_prepare_npu`:快路径条件由 `fused_qkv_a_proj_out.shape[0] < 65535` 追加 `and not nsa_use_prefill_cp(forward_batch)`,nsa prefill CP 时改走非融合的 q/k norm 路径。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅影响 NPU 后端的 DeepSeek V2/DSA + NSA 路径:一是带 prefix 的 nsa cp 请求其 `actual_seq_lengths_kv` 计入 prefix 后数值变大,后端按更长序列处理;二是 nsa prefill CP 下不再走 `fused_split_qk_norm` 快路径,可能带来少量性能开销。无 prefix 场景走 else 分支,行为不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
