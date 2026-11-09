# PR #20308: [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jvlunteren |
| 合并 | 2025-07-07 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/20308 |

## 一句话结论

跳过 causal 全掩 tile，Llama-3.1-8B/H100、bs=1、16K tokens latency 约 1.75x，serving 吞吐 +3%。

## 动机

统一 Triton attention kernel（#16828、#19152）在 prefill 阶段逐 tile 循环、逐 tile 施加 causal mask 并累加。当前实现中最多一半 tile 会被完全掩掉，仍被处理，造成冗余计算，长 prompt 尤其明显。

## 优化点

在 `kernel_unified_attention_2d` 中按当前 q_block 计算任一 query token 所跨越的最长序列前缀 `max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1`，并用 `tl.minimum(..., seq_len)` 修正最后一个 q_block 的 padding，再以 `num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)` 得到需处理的 tile 数；由于 causal masking，超出该前缀的 block 可跳过。其中 `context_len` 为已有上下文长度，`q_block_local_idx * BLOCK_Q` 定位当前 q_block，而 `(BLOCK_M - 1) // num_queries_per_kv + 1` 给出该 q_block 内最后一个 query 的偏移，二者相加即该 q_block 需要覆盖到的最长历史前缀；与真实 `seq_len` 取小可避免末尾 q_block 因 padding 多算 tile。原实现用 `cdiv_fn(seq_len, BLOCK_SIZE)` 对所有 tile 一视同仁地循环，本 PR 只把上界换成 `max_seq_prefix_len`，`for j in range(0, num_blocks)` 的累加逻辑不变。

## 关键改动

```text
vllm/attention/ops/triton_unified_attention.py: kernel_unified_attention_2d 用 max_seq_prefix_len 计算 num_blocks，跳过因果掩掉的 tile。（本次 diff 仅含此 1 个文件）
```

```python
max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
    BLOCK_M - 1) // num_queries_per_kv + 1
max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)

for j in range(0, num_blocks):
```

## 性能声明

模型 `meta-llama/Llama-3.1-8B-Instruct`、NVIDIA H100、`VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_V1=1 python benchmarks/benchmark_latency.py --model meta-llama/Llama-3.1-8B-Instruct --input-len <input-length> --output-len 4 --batch-size <batch-size>`（warmup 10、measure 30 为默认），bs=1、prompt 500/1000/1500/2000/4000/8000/16000：正文称 16K tokens、bs=1 时 Triton Unified Attention Kernel 约 1.75x。`benchmark_serving.py`（sharegpt，仅含 <2000 tokens 序列）：984 请求下 Request throughput 44.37→45.90 req/s、Output tok/s 8793.44→9137.19、Total tok/s 18297.62→18954.74；Mean TTFT 3874.12→3588.36 ms、Mean TPOT 88.60→83.17 ms、Mean ITL 40.26→38.12 ms；正文称吞吐约 +3%、TTFT/TPOT/ITL 等延迟指标降超 5%。benchmark_latency 的结果以图形式给出（PR 中的 benchmark_latency_results 图），正文未列出逐点数值表。

## 验证与回归风险

正确性用 `VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 500`：V1 FlashAttention gsm8k flexible-extract 0.794±0.0181 / strict-match 0.772±0.0188；本 PR（TRITON_ATTN_VLLM_V1）0.800±0.0179 / 0.784±0.0184。可能影响路径：TRITON_ATTN_VLLM_V1 后端的 prefill，尤其是长 prompt 与 causal mask 边界（末尾 q_block padding）；若 `max_seq_prefix_len` 估计偏小会漏算有效 tile，偏大则退化回原来的多余计算，正文以 gsm8k 精度对照验证数值一致性。正文未给出新增单测命令或测试通过数。

## 关联

#16828、#19152（统一 Triton attention kernel 的前序 PR）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
