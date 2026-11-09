# PR #20411: [Nvidia] Integrate SM100 cudnn prefill API to MLA prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elfiegg |
| 合并 | 2025-07-16 |
| 分类 | Attention / Kernel |
| Label | performance, ready, v1, deepseek |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),实际改动集中在 envs.py 与 mla/common.py |
| 链接 | https://github.com/vllm-project/vllm/pull/20411 |

## 一句话结论

在 SM100 上把 cuDNN prefill attention API 接入 MLA prefill 路径(环境开关 `VLLM_USE_CUDNN_PREFILL=1`),作为 Blackwell 上 FlashInfer prefill 之外的又一选择。

## 动机

Blackwell 上 MLA prefill 此前默认走 FlashInfer ragged prefill(快于 FA2),FA 作为兜底。NVIDIA 提供了 cuDNN batch prefill attention API(`cudnn_batch_prefill_with_kv_cache`),针对长输入 prefill(如 DeepSeek 10k input)可能有额外收益;本 PR 将其接入 MLA common 后端,允许在两种实现间切换。

## 优化点

新增环境开关 `VLLM_USE_CUDNN_PREFILL` 与 `use_cudnn_prefill()`(要求 flashinfer 可用且 SM100+);启用时 `MLACommonMetadataBuilder` 分配每序列 `CUDNN_WORKSPACE_SIZE=12800` 字节 * max_num_seqs 的 int8 workspace,并改用新的 `CudnnPrefillMetadata`(带 `query_seq_lens` 与 `cudnn_workspace`,chunked context 里补 `seq_lens` 张量)。`MLACommonImpl` 的 prefill 函数指针分派到两个 cudnn 实现:`_run_prefill_new_tokens_cudnn` 以 causal=True 调 `cudnn_batch_prefill_with_kv_cache`(q/kv 实际长度都取 query len,`is_cuda_graph_compatible=True`,`return_lse=True`);`_run_prefill_context_chunk_cudnn` 处理 chunked prefill 的 context chunk(causal=False,kv 长度取该 chunk 的累计 seq_lens)。chunked context metadata 构建时统一附带 `seq_lens`。flashinfer prefill 优先级不变:cudnn 开启时 flashinfer prefill 被关闭,decode 路径不受影响。

## 关键改动

```text
vllm/envs.py: 新增 VLLM_USE_CUDNN_PREFILL 开关
vllm/v1/attention/backends/mla/common.py: CudnnPrefillMetadata、workspace 分配与两个 cudnn prefill 实现
依赖:flashinfer commit 9157d0514(TORCH_CUDA_ARCH_LIST="10.0a" 构建)
```

## 性能声明

PR 正文仅附一张性能对比截图(具体数字在图片中,素材文本未包含),复现命令为 DeepSeek-Coder-V2-Lite-Instruct FP8、chunked prefill、input-len 10000 / output-len 1000 / num-prompts 100 的 offline throughput benchmark,`VLLM_USE_CUDNN_PREFILL=1`。PR 未注明可引用的量化数字与硬件(依上下文为 SM100)。

## 验证与回归风险

- diff 中未见新增测试;验证依赖 MLA 既有测试与上述 benchmark。
- 风险:`max_sequence_kv` 在 new-token 路径取 `max_query_len`(即 kv 与 query 等长的 KV-cache 布局假设,仅适用 MLA 紧凑 KV);`return_lse=False` 不支持(强制 True);cudnn workspace 随 max_num_seqs 线性增长;开关默认关闭,不影响默认 FlashInfer/FA 路径;与 `_pad_v=False` 的组合要求 KV 无 padding。

## 关联

- 同区域:#20034(flashinfer ragged prefill for MLA,即被 cudnn 关闭的默认路径)、#20769(SM100 CUTLASS MLA decode)、#21078(flashinfer trtllm-gen decode)、#34732(FA4 MLA prefill,后续)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
