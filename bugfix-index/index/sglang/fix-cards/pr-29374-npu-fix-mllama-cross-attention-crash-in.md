# PR #29374: [NPU] Fix mllama cross-attention crash in ascend extend SDPA

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29374](https://github.com/sgl-project/sglang/pull/29374) |
| 作者 | syy-hw |
| 合并日期 | 2026-06-26 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
PR #26147 的 SWA 改动把冗余 query 张量尺寸从文本序列长度改为 KV 窗口宽,破坏了 mllama cross-attention(Q=16 与 KV=6404 长度不等),Ascend NPU 上服务器无法启动;按场景分支取尺寸修复。

## 问题现象
Ascend NPU 后端上,mllama(Llama-3.2-11B-Vision-Instruct)的 cross-attention 在 extend SDPA 路径崩溃,导致服务器无法启动。正文给出根因数据:cross-attention 中 Q 来自文本(16 tokens)而 KV 来自 encoder/图像(6404 tokens),冗余 Q 张量被按 encoder 长度(6404)分配但实际只有 16 个真实 query,赋值时 shape mismatch 失败。

## 根因
正文披露:PR #26147(Gemma4 SWA support)引入回归——`run_sdpa_forward_extend` 中冗余 query 张量的尺寸从 `seq_lens[seq_idx]`(文本序列长度)改为 `atten_end_kv - atten_start_kv`(KV 窗口宽)。该改动对 SWA self-attention 正确,但破坏 cross-attention:Q 与 KV 长度本就合法地不等,窗口宽尺寸导致赋值 shape mismatch。

## 修复方案
改 `python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py` 的 `run_sdpa_forward_extend`:

- 提取 `is_swa_self_attn` 标志(`sliding_window_size > -1 and encoder_lens is None`),消除重复条件判断。
- 冗余 query 尺寸按场景分支:SWA self-attention 取 `redundant_len = atten_end_kv - atten_start_kv`(裁剪后窗口宽,保持 Q.len == K.len 供 causal mask 使用),`query_start_idx = max(prefill_seq_len_q - atten_start_kv, 0)`;cross-attention / 非 SWA self-attention 恢复 `redundant_len = int(seq_lens[seq_idx].item())`(原文本序列长度语义),`query_start_idx = prefill_seq_len_q`。

```diff
+            if is_swa_self_attn:
+                redundant_len = atten_end_kv - atten_start_kv
+                query_start_idx = max(prefill_seq_len_q - atten_start_kv, 0)
+            else:
+                redundant_len = int(seq_lens[seq_idx].item())
+                query_start_idx = prefill_seq_len_q
             per_req_query_redundant = torch.zeros(
-                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
+                (per_req_query.shape[0], redundant_len, per_req_query.shape[2]),
```

验证:Llama-3.2-11B-Vision-Instruct 在 NPU 上跑 MMMU,score 0.3301、总延迟 165.395s、输出吞吐 18.036 token/s,测试通过。

## 影响与回归风险
- 影响窗口:自 PR #26147 合并引入回归起,至本 PR 合并(2026-06-26);正文未给出版本号。
- 回归面:仅 NPU `ascend_torch_native_backend.py` 的 extend SDPA 路径。Gemma4 SWA self-attention 行为不变(分支取值与改前相同);mllama cross-attention 与非 SWA self-attention 恢复原尺寸语义。非 NPU 后端无改动。

## 关联
- PR #26147(引入回归): https://github.com/sgl-project/sglang/pull/26147

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
