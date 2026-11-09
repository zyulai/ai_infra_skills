# PR #62029: [Attention] Use trtllm-gen for the FlashInfer DCP prefill context pass

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @GagandeepKang-cohere |
| 合并 | 2026-10-20 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, nvidia, verified |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/62029 |

## 一句话结论

DCP prefill context pass 改跑 trtllm-gen context kernel,8xB200 100k 输入 TTFT 降低 39-48%,输出吞吐 1.23-1.79x。

## 动机

FlashInfer 的 decode-context-parallel (DCP) prefill 一直跑 native FlashInfer context kernel,而非 DCP prefill 在满足条件时已自动选择更快的 trtllm-gen kernel;DCP 与非 DCP 路径的 kernel 选择不一致。trtllm-gen 的 non-causal + LSE 输出能力(flashinfer#3020、#3116)与 DCP 支持(#25438)同期但此前未接到 DCP context pass 上。本 PR 在与非 DCP prefill 相同的适用条件下,把 DCP context pass 切到 trtllm-gen;#58519 是围绕同一 pass 减少周边开销的互补工作。

## 优化点

新增 `TrtllmDCPContextWrapper` 作为 native `BatchPrefillWithPagedKVCacheWrapper` 的替身:调用 `trtllm_batch_context_with_kv_cache(causal=False, return_lse=True)` 于 DCP-local paged KV;其 LSE 与 native kernel 同为 base 2,故 combine/merge 逻辑不变。启用条件与非 DCP prefill 的 trtllm-gen 自动选择一致:SM100/SM103 且 `use_trtllm_attention` 非 0、KV cache 未量化、无 sliding window、无 soft-capping;不满足的层保留 native kernel。关键边界处理:没有本地 context 的请求(kv_len==0)trtllm-gen 不写输出与 LSE,workspace 中的陈旧 softmax stats 会间歇性破坏 merge——这些行在 combine 中按空 shard 掩掉(`mask_dcp_empty_shards_`),输出因 all-gather combine 乘 0 权重而需零初始化,LSE 置 -inf。workspace 适配检查:`fits_workspace` 镜像 trtllm-gen launcher 的 LSE stats 划痕公式(heads x batch x round_up(max_q_len,256) 个 float2 + 1 MiB),超出则该步回退 native context pass。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: TrtllmDCPContextWrapper(plan/run/fits_workspace)与 BatchDCPPrefillWrapper 的 use_trtllm_context 选择逻辑
tests/kernels/attention/test_flashinfer_trtllm_attention.py: test_flashinfer_dcp_prefill_trtllm_context(与 native 对照)与 workspace 回退测试
vllm/v1/attention/ops/dcp.py: mask_dcp_empty_shards_(空 shard 掩蔽,diff 省略)
```

## 性能声明

负载:`vllm bench serve --dataset-name random`,100k 输入 / 512 输出,`--request-rate inf --ignore-eos`,prefix caching off,a2a DCP combine,每 arm 单次运行。Qwen3-30B-A3B,8xB200,TP8/DCP2/EP8,MRCR/GSM8K 用真实权重:mean TTFT 2190→1332 ms (c=1) 至 16277→8460 ms (c=32),即 -39-48%;输出吞吐 106.9→131.5 tok/s (1.23x) 至 250.7→448.8 tok/s (1.79x),c1-c32 区间 1.23-1.79x;TPOT 在高并发改善仅因 decode 步与更短的 prefill chunk 共批。GLM-4-9B-chat (32 Q / 2 KV heads),4xB200,TP4/DCP2:mean TTFT -37-44%(3220→2019 至 33043→18366 ms),输出吞吐 1.26-1.68x(90.3→113.9 至 168.1→282.6 tok/s)。精度:MRCR match ratio (40 samples) native 0.480 vs trtllm-gen 0.496;GSM8K (1319, 5-shot) 0.888 vs 0.899——均在噪声内;kernel 级对照(全层全 rank、真实 metadata)最大相对输出误差 2.3e-4,LSE 匹配。

## 验证与回归风险

测试:新增 `test_flashinfer_dcp_prefill_trtllm_context`(num_heads (8,1)/(16,2)、dcp_world_size 2/8、head_size 64/128/256、block_size 16/64、含 [0,37,1000,0,4096] 与 [0,0] 两种 context 组合,断言无本地 context 行输出为 0、LSE 为 -inf,有 context 行与 native 对照 atol/rtol 1e-2),加入 "(B200) Kernels" CI;`test_trtllm_dcp_context_workspace_fit` 验证 LSE scratch 超限回退 native。PR 注明 `pytest ... -k dcp_prefill` 在 B200 上为 TODO 状态。风险面:FP8 KV 仍走 native kernel(且 FlashInfer DCP + FP8 KV 在 SM100 上启动即失败,与本 PR 无关、另行修复);`use_trtllm_attention=0` 会同时关闭 trtllm-gen decode 与本路径;GQA DCP e2e 测试(test_context_parallel.py)需 4 GPU 且仅 SM90,不覆盖 Blackwell;空 shard 掩蔽逻辑与 trtllm DCP decode 路径保持一致。

## 关联

同区域:#25438(FlashInfer DCP 原始支持)、#58519(同 pass 周边开销削减,互补)、flashinfer-ai/flashinfer#3020 与 #3116(non-causal + LSE kernel 能力)、trtllm-gen 非 DCP prefill 自动选择路径。pack 素材未列出其他明确前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
