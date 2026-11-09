# PR #9675: Fix TRTLLM MLA Cuda KV Blocks Causing accuracy drop

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9675](https://github.com/sgl-project/sglang/pull/9675) |
| 作者 | farazkh80 |
| 合并日期 | 2025-08-30 |
| labels | (none) |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
trtllm_mla 解码 CUDA graph 捕获时 KV 索引宽度按当批最大序列长度分配,replay 遇更长序列精度下降;改为按 max_context_len 全宽分配并记录真实 max_seq_len。

## 问题现象
DeepSeek-R1-0528-FP4 + `--attention-backend trtllm_mla --kv-cache-dtype fp8_e4m3`(TP4+EP+DP attention)长上下文精度下降:Math 500 在「TP EP DP 命令、16k tokens + cuda graph ON」时 pass@1 仅 84.60%(no_answer 15.00%,正文标注 **CONCERNING**);对照 TP only 32k(cuda graph ON)为 97.60%、16k FP8 97.60%。修复后给出 GPQA 76.26%、GSM8k 94.7%、MMLU Pro 81.91%、Math 500 32k trtllm_mla BF16 97.20% 等结果(其中 Math 500 因 issue #9704 加了 `--disable-chunked-prefix-cache`)。

## 根因
正文未披露详细机制,仅说明修法是「按 `self.max_context_len` 设置 kv_indices 数量」。由 diff 推断:CUDA graph 捕获时 `block_kv_indices` 宽度按捕获当批的 `seq_lens.max()` 算(`max_seqlen_pad`),replay 时若实际序列比捕获时更长,KV 块索引超出预分配宽度,导致长上下文下注意力看不到完整 KV,造成精度下降;另外 `forward_decode` 里 `max_seq_len=int(metadata.block_kv_indices.shape[1] * self.page_size)` 用的是索引表宽度而非真实最大序列长度。

## 修复方案
`python/sglang/srt/layers/attention/trtllm_mla_backend.py`:
1. CUDA graph 捕获改为全宽:`max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)`,保证 replay 时更长序列安全;
2. `TRTLLMMLADecodeMetadata` 新增 `max_seq_len` 字段,capture 时记录 `int(seq_lens.max().item())`,replay 时更新(优先用 `seq_lens_cpu` 避免 GPU 同步),非 graph 路径 `init_forward_metadata` 同样记录;
3. `forward_decode` 改传 `max_seq_len=metadata.max_seq_len`(kernel 需要 int 而非 tensor);
4. 输出不再按 `v_head_dim` 切片,直接 `raw_out.view(-1, ...)`(注释:Reshape output directly without slicing)。
并更新 `python/sglang/test/attention/test_trtllm_mla_backend.py` 相关断言。

```diff
-        max_seqlen_pad = self._calc_padded_blocks(seq_lens.max().item())
-        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_seqlen_pad]
+        # Capture with full width so future longer sequences are safe during replay
+        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
+        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]
```

## 影响与回归风险
- 影响窗口:正文未说明(触发条件为 trtllm_mla 后端 + CUDA graph 解码 + replay 时序列长于捕获时)。
- 回归面:trtllm_mla 解码的 CUDA graph 捕获/重放与非 graph 路径的 metadata 构造与 `max_seq_len` 传参;KV 索引按 max_context_len 全宽分配会增大索引表内存占用;输出 reshape 不再切片,若模型的 `v_head_dim` 与输出切片假设相关,该路径行为改变(正文未就此展开)。

## 关联
- https://github.com/sgl-project/sglang/pull/8638#issuecomment-3193478283 (fix 对象)
- https://github.com/sgl-project/sglang/issues/9704 (chunked prefix cache bug,评测时绕过)
- https://github.com/sgl-project/sglang/pull/8638 (基准对照)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
