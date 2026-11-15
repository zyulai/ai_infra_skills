# PR #66018: [Attention][MiniMax-M3] Add FlashInfer MSA backend for SM90 and SM120/SM121

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rmhaskarnvidia |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready, nvidia, verified, minimax |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66018 |

## 一句话结论

为 MiniMax-M3 sparse attention(lightning indexer + block-sparse attend)新增 FlashInfer `msa_ops` 后端,覆盖 SM90 与 SM120/SM121(此前这些 GPU 只能跑 Triton fallback,vendored MSA kernel 是 SM100-only):H100 kernel 层 decode 1.02–2.13x、prefill 2.6–5.6x、mixed 3.5–5.0x,E2E ITL p50 低 9–22%。

## 动机

MiniMax-M3 sparse attention 在 SM90 与消费级 Blackwell(SM120/SM121)上只能跑 Triton fallback,vendored MSA kernel 仅支持 SM100。本 PR 取代 #48994(仅 SM120/SM121,其 commits 保留)并新增 SM90。需要 FlashInfer 0.7.3(#65930 的 pin bump);更老的 FlashInfer 上 feature probe 保持 Triton 路径,合并顺序无关。

## 优化点

- 单一选择谓词:`minimax_m3_use_flashinfer_msa()` 每 layer 计算一次,同时喂给 indexer 与 attend 两个 selector;FlashInfer 两个 impl 共享 top-k buffer(布局与 Triton 对不同),混合配对必须永不发生。
- Indexer:整个 batch 一次 `msa_proxy_score` + 一次 `msa_topk_select`(无 decode/prefill 拆分);每 token 的 causal extent 与 trailing local window 经 per-token `num_valid_pages` / `force_end_blocks` 直接进 top-k kernel,Python 侧无需逐 token masking。
- Attend:kernel 原地读 vLLM 的 paged KV cache(从 packed cache split 出 K/V view);fp8 descales 精确折进 softmax scale 与输出 scale;SM90 上 kernel 写入 `out=`,backend 请求其读取的 head-major `LBHNC` 布局。
- SM90 支持:fp8 E4M3 main + index cache、bf16 activation、top-k 16、GQA group 8 或 16(MiniMax-M3 在 TP1–TP8)、每 rank 1/2/4 index head;其余一切留在 Triton。
- Feature probe:`vllm/utils/flashinfer.py` 的 `has_flashinfer_msa*` 逐项探测所需能力;FlashInfer 未暴露 SM90 支持与 per-token top-k 的 flag,这两项从 op 签名探测。
- SM90 workaround:FlashInfer 0.7.3 的 `msa_proxy_score` 在 prefill 步可能越界读 `q` 末尾至多 127 行(可能 fault),indexer 在 SM90 prefill 步 pad `q`(每 prefill 步一次小 copy),待 FlashInfer 修复后移除。

## 关键改动

> pack 的 diff 被截断(2 个文件未展示,`sparse_attention_flashinfer.py` 亦在展示中途截断),以下仅覆盖已展示部分。

```text
vllm/models/minimax_m3/common/sparse_attention.py: 新增 minimax_m3_use_flashinfer_msa 门控;select_main_backend_and_impl_cls 返回 FlashInfer 后端/impl
vllm/models/minimax_m3/common/indexer.py: select_indexer_impl_cls 接受 use_flashinfer_msa,选 MiniMaxM3IndexerFlashInferImpl
vllm/models/minimax_m3/nvidia/indexer_flashinfer.py(新增): 整 batch msa_proxy_score + msa_topk_select,per-token num_valid_pages/force_end_blocks
vllm/models/minimax_m3/nvidia/sparse_attention_flashinfer.py(新增): msa_sparse_attention / msa_sparse_decode_attention attend(SM90 LBHNC + out=)
vllm/models/minimax_m3/nvidia/model.py: 每 layer 一次计算 use_flashinfer_msa 并传给两个 selector
tests/kernels/attention/test_minimax_m3.py: FlashInfer indexer/attend 一致性测试 + SM90/SM12x gate 测试
```

```python
if current_platform.is_device_capability(90):
    # FlashInfer's SM90 MSA ops take fp8 E4M3 main and index caches with
    # bf16 queries, GQA groups of 8 or 16, and 1, 2 or 4 index heads.
    return (
        kv_cache_dtype in ("fp8", "fp8_e4m3")
        and indexer_kv_dtype in ("fp8", "fp8_e4m3")
        and model_dtype == torch.bfloat16
        and num_heads % num_kv_heads == 0
        and num_heads // num_kv_heads in (8, 16)
        and num_kv_heads in (1, 2, 4)
        and has_flashinfer_msa_sm90()
        and has_flashinfer_msa_per_token_topk()
    )
```

## 性能声明

- Kernel 层(单张 H100 80GB,每 TP8 rank 一个 MiniMax-M3 sparse layer = indexer(proxy score + top-k)+ attend;8/1/1 heads,head dim 128,fp8 E4M3 KV 与 index cache,page 128,top-k blocks 16 / local window 1,max model len 1,048,576):decode(CUDA graph replay,4 query tokens/请求 MTP verify)一个 sparse layer 1.02–2.13x 快、从不更慢——例:1 请求 8k 1.42x(19.4→13.6 µs)、48 请求 8k 1.85x(113.6→61.5 µs)、1 请求 128k 2.06x(33.4→16.2 µs)、48 请求 1M 1.09x(2308.4→2118.0 µs);prefill(eager)2.6–5.6x——1x8192 无 prefix 5.64x(1693.7→300.3 µs)、+1016k cached prefix 4.32x(13050.5→3024.0 µs);mixed decode+prefill 3.5–5.0x——8 decode(4 tokens @128k)+ 1x2048 prefill 4.97x(1668.0→335.4 µs)。
- 端到端(8x H100 与 8x H200,MiniMax-M3.1-Flash TP8,fp8 KV 与 index cache,MTP 3 draft tokens,prefix caching,agentic multi-turn traces,concurrency 1–10,对照同 build 关闭 FlashInfer 路径):每个并发点上 ITL p50 低 9–22%,TTFT p50 低 3–19%(两张 GPU 均是);GSM8K(400 题)噪声内不变:96.75% vs 96.25%(H100),96.50% vs 97.25%(H200)。
- 数值:与 Triton 对在相同输入上 attend rel-L2 ≤ 3.4e-3;top-k 行 decode 87.5–100% 一致、prefill ≥ 99.4%(差异为等分 blocks 的 tie)。

## 验证与回归风险

- `pytest tests/kernels/attention/test_minimax_m3.py`(H100,PyPI FlashInfer 0.7.3):`-k flashinfer` 11 passed、11 skipped(bf16 cache 与 NHD 布局,SM90 不服务);整文件 129 passed、98 skipped(其他架构)、0 failed。
- SM120/SM121 parity 见 #48994(RTX 5080 验证)。kernel 层与 E2E 用 FlashInfer source build,其 SM90 `msa_ops` 源码与 0.7.3 release 相同。
- CI:新测试进既有 kernel-attention job,但仅在 SM90/SM120/SM121 且 FlashInfer ≥ 0.7.3 上运行(否则 skip);selection-predicate 测试 CPU-only 恒跑。
- 风险/未包含:SM90 依赖 q pad workaround(FlashInfer 修复前每 prefill 步一次小 copy);未做 FlashInfer SM90 top-k JIT kernel 预热——新 shape 的首个 prefill/mixed 步要现场编译一次(1–2 s,之后走 FlashInfer disk cache;唯一预热 hook 是 FlashInfer 私有 API,留给后续);SM90 仅接受 fp8 E4M3 cache。

## 关联

- #48994(被本 PR 取代,SM120/SM121 部分,commits 保留);#65930(FlashInfer 0.7.3 pin bump)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
