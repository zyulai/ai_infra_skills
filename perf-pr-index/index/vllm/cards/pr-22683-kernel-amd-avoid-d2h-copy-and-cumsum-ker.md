# PR #22683: [Kernel][AMD] Avoid D2H copy and cumsum kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mxz297 |
| 合并 | 2025-08-12 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, rocm, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/22683 |

## 一句话结论

AITER prefill attention 去掉每层 D2H 拷贝与 cumsum kernel，prefill latency 改善约 10%。

## 动机

PR 正文说明：当前 AITER attention 的 prefill 会引入一次 D2H copy（造成 CPU 与 GPU 之间的同步）和一次 cumsum kernel。D2H copy 用于在 host 侧取得实际 KV token 数以便分配中间 buffer，而该值在推理过程中不会变化，因此可以在 AttentionMetadata 构造时预先计算；cumsum kernel 同理可提前到构造阶段完成。正文称改后仍可看到 memcpy 与 GPU bubbles，但观察到约 10% prefill latency 改善。

## 优化点

改动集中在 `vllm/v1/attention/backends/rocm_aiter_fa.py`。`AiterFlashAttentionMetadata` 新增 `num_actual_kv_tokens: int` 与 `cu_seq_lens: Optional[torch.Tensor]` 字段。在 `build()` 中，当 `max_query_len > 1` 时一次性构造 int32 的 `cu_seq_lens`（`torch.cumsum(seq_lens, out=cu_seq_lens[1:])`）并算得 `num_actual_kv_tokens = int(cu_seq_lens[-1].item())`，decode 路径（`max_query_len <= 1`）置 `cu_seq_lens=None`、`num_actual_kv_tokens=0`。`forward()` 中删除原来每层 prefill 都执行的 `torch.zeros` + `torch.cumsum`，改为直接使用 `attn_metadata.cu_seq_lens` 与 `attn_metadata.num_actual_kv_tokens` 传给 `flash_attn_varlen_func` 的 `cu_seqlens_k` 与 `total_tokens`。

## 关键改动

```text
vllm/v1/attention/backends/rocm_aiter_fa.py: 把 KV token 数与 cu_seq_lens 的 cumsum 预计算提到 AttentionMetadata.build，forward 复用，消除逐层 D2H 同步与 cumsum kernel
```

```text
+    num_actual_kv_tokens: int
+    cu_seq_lens: Optional[torch.Tensor]
+        if max_query_len > 1:
+            cu_seq_lens = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32,
+                                      device=seq_lens.device)
+            torch.cumsum(seq_lens, dim=0, dtype=cu_seq_lens.dtype,
+                         out=cu_seq_lens[1:])
+            num_actual_kv_tokens = int(cu_seq_lens[-1].item())
...
-                    cu_seqlens_k=cu_seq_lens,
+                    cu_seqlens_k=attn_metadata.cu_seq_lens,
-                    total_tokens=attn_metadata.total_tokens,
+                    total_tokens=attn_metadata.num_actual_kv_tokens,
```

## 性能声明

正文仅给出 "Saw about 10% prefill latency improvement"，未注明模型、硬件、精度、workload 等测试条件，也未提供具体延迟数值或基准表格（性能相关截图为图片链接）。PR 未注明测试条件。

## 验证与回归风险

正文的 Purpose、Test Plan、Test Result 章节均为空标题，未给出测试命令、用例或新增测试；Checklist 四项均未勾选。改动仅作用于 ROCm AITER FA backend 的 prefill（`max_seqlen_q > 1`）路径，decode 分支使用 `cu_seq_lens=None`、`num_actual_kv_tokens=0`；正文未说明对其他模型、硬件或精度路径的回归影响。正文另附两张说明改动前后 memcpy 与 GPU bubble 的截图链接。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
