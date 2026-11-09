# PR #48770: [2/N][Attention] Enable masked MHA for sparse MLA prefills

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-07-31 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready, ci/build, v1 |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 另有 3 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/48770 |

## 一句话结论

为 sparse MLA prefill 增加 masked MHA 路径(FA4 + bit-packed top-k 掩码):短序列 prefill 不再退回 sparse MQA,延续 DeepSeek V3.2 论文的 DSA 模拟优化,补齐 #47327 dense MHA shortcut 的中间区间。

## 动机

DeepSeek V3.2 论文(arXiv 2512.02556)指出:短序列 prefill 下用 masked MHA 模拟 DSA 比 sparse MQA 效率更高("for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions")。#47327 已实现 `seq_len < topk` 时走 dense MHA 的 shortcut;本 PR 把这一优化扩展到 `seq_len > topk` 的区间——用只 attend 各 query token top-k 集合的 masked MHA 替代 sparse MQA。

## 优化点

可用性判定 `_is_masked_mha_available`:仅限 CC 10.x(Blackwell)、DeepSeek-V3.2 精确形状(128 heads、kv_lora_rank 512、qk_nope 128、qk_rope 64、v 128)、FA4、非量化 KV cache。路由 `forward_impl` 按 backend × TP × (query_len, seq_len bucket) 查 `_DSV32_MASKED_MHA_THRESHOLDS` 阈值表(如 FLASHMLA_SPARSE TP8:seq≤2048 需 q≥512、≤8192 需 q≥1024、≤32768 需 q≥16384;FLASHINFER_MLA_SPARSE 有 TP8 一组),满足才切 MHA,受 `sparse_mla_force_mqa` 覆盖。掩码构建:`_build_topk_mask` 用两个 Triton kernel 把每 token 的 top-k 索引以 `atomic_or` scatter 成 int32 bit-packed 掩码,在 64 MiB workspace 预算内构建全序列全局掩码并跨 suffix 与所有 context chunk 复用(免 per-chunk 重建),超预算回退 per-chunk 索引 remap。执行 `_run_masked_mha` 调 `flash_attn_varlen_func`(FA4,mask_mod + aux_tensors),chunked prefill 各 chunk 用 `merge_attn_states` 合并。依赖 vllm-project/flash-attention#155(cmake GIT_TAG 更新)。

## 关键改动

```text
vllm/model_executor/layers/attention/sparse_mla_attention.py: masked MHA 可用性判定、top-k bitmask Triton 构建、forward_mha/chunked context 执行
vllm/model_executor/layers/attention/mla_attention.py: 阈值表路由 _use_masked_mha;ChunkedContextMetadata 增加 context_lens/query_lens_cpu/topk_mask_workspace
cmake/external_projects/vllm_flash_attn.cmake: vllm-flash-attn 指向含 masked MHA 的 commit
tests/v1/attention/test_sparse_mla_backends.py, test_sparse_mla_mask.py: masked MHA prefill 正确性与掩码构建一致性
benchmarks/attention_benchmarks/: masked_mha variant、crossover sweep 配置
```

## 性能声明

PR 正文仅附三张结果图(mha_vs_mqa:num_q_heads=128 与 TP8/16 heads 对比图、TP1/2/4/8 speedup heatmap),未给出具体数字,硬件未注明(代码路径要求 Blackwell/FA4)。定性结论:低于阈值的短序列 prefill 中 masked MHA 快于 sparse MQA;新配置 `mla_sparse_masked_mha_vs_mqa.yaml`(DeepSeek-V3 形状,q32-q65536 共约 90 个 spec)用于绘制 crossover 曲线,dense MHA 上限 25349、masked MHA 上限 8192。

## 验证与回归风险

`tests/v1/attention/test_sparse_mla_backends.py` 新增 masked_mha 与 masked_mha_chunked_context prefill 正确性用例(topk=200,随机 top-k 索引与 per-token SDPA 参考对比);`tests/v1/attention/test_sparse_mla_mask.py` 验证 `_build_topk_mask` 单请求与通用路径结果一致。回归影响:DeepSeek-V3.2 形状 + Blackwell + FA4 + 非 KV 量化的 prefill 路由变化;chunked context 元数据新增字段;DCP > 1 时 masked MHA with context 抛 NotImplementedError;依赖 vllm-flash-attn 版本升级。

## 关联

#47327(前序 1/N:dense MHA shortcut,`seq_len < topk`);依赖 vllm-project/flash-attention#155;同区域 #33451(FlashInfer sparse MLA backend,阈值表中含 FLASHINFER_MLA_SPARSE)、#52212(ROCm sparse-MLA decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
