# PR #14431: [Kernel] [V1] Further optimizations to ROCm (Triton) Backend to better handle GQA.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tdoublep |
| 合并 | 2025-03-14 |
| 分类 | Attention / Kernel |
| Label | ready |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 集中于 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/14431 |

## 一句话结论

重写 Triton `chunked_prefill_paged_decode` 的 GQA 处理,一个 program 处理一个 KV head 的全部 query head,llama3.1-8B 吞吐 +25%。

## 动机

V1 的 Triton attention 后端中,`kernel_paged_attention_2d` 以 `(num_seqs, num_query_heads)` 为 grid,每个 program 只处理一个 query head,并用标量内积 `tl.sum(K * Q)` 计算 attention score。GQA 模型下同一 KV head 的多个 query head 各自独立读 K/V cache,K/V 带宽被重复消耗,导致 Triton 实现与 V1 CUDA 后端 (FlashAttention) 差距明显。本 PR 让每个 program 覆盖一个 KV head 的全部 query head,以 `tl.dot` 走矩阵化路径。

## 优化点

grid 从 `(num_seqs, num_query_heads)` 改为 `(num_seqs, num_kv_heads)`,`query_head_idx` 由 `kv_head_idx * num_queries_per_kv + tl.arange(0, num_queries_per_kv_padded)` 生成(pad 到 next_power_of_2 且至少 16,越界用 `head_mask` 掩掉)。Q 载入为 `(num_queries_per_kv, HEAD_SIZE)` 二维块,`S += scale * tl.dot(Q, K)` 一次算出整组 score;softmax 的 M/L/acc 状态全部升级为按 query 行的向量/矩阵;V 按 `(BLOCK_SIZE, HEAD_SIZE)` 载入,`acc += tl.dot(P.to(V.dtype), V)`。alibi、sliding window、fp8 KV cache 分支同步适配二维形状。KV cache 对同一 KV head 只载入一次,供组内全部 query head 复用。

## 关键改动

```text
vllm/attention/ops/chunked_prefill_paged_decode.py: kernel_paged_attention_2d GQA 二维化重写 + launcher grid/参数调整
```

```python
seq_idx = tl.program_id(0)
kv_head_idx = tl.program_id(1)
...
query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
    0, num_queries_per_kv_padded)
head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
head_mask = head_mask & (query_head_idx < num_query_heads)
...
# S : (num_queries_per_kv, BLOCK_SIZE,)
S = tl.where(head_mask[:, None] & seq_mask, 0.0, float("-inf")).to(tl.float32)
S += scale * tl.dot(Q, K)
```

## 性能声明

Serving benchmark(V1,benchmark_serving,ShareGPT,1000 请求,Llama-3.1-8B-Instruct;PR 明确写 H100 作为对比平台,`ROCmAttentionBackend` 亦运行于 H100):

- main 的 `ROCmAttentionBackend`:req/s 35.44,output tok/s 6991.12,Mean ITL 51.02 ms;
- 本 PR 的 `ROCmAttentionBackend`:req/s 44.19(+24.7%),output tok/s 8711.51,Mean ITL 39.51 ms——即 TLDR 所称 25% 吞吐提升;
- main 的 `FlashAttentionBackend`(V1 CUDA):req/s 48.15,output tok/s 9534.13;改后 Triton 落后该基线约 8%。

Mean TTFT 3947.90 ms(略高于基线 3483.50 ms),Median TPOT 50.22 ms(基线 63.35 ms)。

## 验证与回归风险

PR 正文未列出专门的单元测试,验证以上述 serving benchmark 为主。风险面:`num_queries_per_kv_padded = max(next_power_of_2(num_queries_per_kv), 16)` 引入 padded lane,依赖 `head_mask` 正确屏蔽;alibi/sliding window/fp8 cache 各分支均被改写,非 GQA(MQA/MHA,queries_per_kv=1)与极端 head 配置需回归;该 op 为 Triton 后端 chunked prefill + paged decode 共用路径,影响面集中在 ROCm/Triton attention 后端。

## 关联

同区域(ROCm/Triton attention):索引内 PR #12348(ROCm 更快 custom paged attention kernel)、PR #34709(ROCm WvSplitK skinny GEMM)、PR #50578(ROCm MLA asm decode);本文件同 op 的后续演进可对照 V1 CUDA 后端卡 PR #13747(FlashMLA)系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
