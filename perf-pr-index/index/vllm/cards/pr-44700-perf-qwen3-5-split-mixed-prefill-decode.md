# PR #44700: [PERF] [Qwen3.5] Split mixed prefill+decode batches: route decodes to the recurrent kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vadiklyutiy |
| 合并 | 2026-06-06 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, v1, qwen |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 4 个文件(qwen_gdn_linear_attn.py、gdn_attn.py、新增 split 集成测试与 MTP eval 配置) |
| 链接 | https://github.com/vllm-project/vllm/pull/44700 |

## 一句话结论

Qwen3.5 GDN 混合 batch 拆分:decode 剥离到 recurrent kernel、prefill 尾部走 chunk kernel,消除 64-token 空转 chunk,混合场景 kernel 提速 1.93x、E2E 吞吐 +24.4%。

## 动机

GDN attention 之前把所有 non-spec token(包括 1-token decode)统一送入 `chunk_gated_delta_rule`。每个 decode 会成为独立 chunk 并被 pad 到 `FLA_CHUNK_SIZE=64`,因此 D 个 decode 意味着 D 个近空的 64-token chunk 的浪费计算。实测(B200):decode 953 走 fused_sigmoid 1843.7µs + prefill 8192 走 CuteDSL chunk 464.1µs,合计 2370.8µs;而 953 decode + 7239 prefill 混合统一走 chunk kernel 要 4361.6µs——比分别执行还慢。MTP(spec)路径此前已经按类似方式处理。

## 优化点

在 `GatedDeltaNet._forward_core` 中,当 non-spec batch 混合 prefill 与 decode(`spec_sequence_masks is None` 且两者均大于 0)时,把 batch 前部的连续 decode 前缀(decode-first front slice)剥离给 recurrent update kernel `fused_sigmoid_gating_delta_rule_update`(即 spec 路径所用),只把 prefill 尾部交给 chunk kernel。metadata builder(`GDNAttentionMetadataBuilder`)预计算并 rebases chunk 元数据:`prefill_query_start_loc`/`prefill_state_indices`/`prefill_has_initial_state` 基于去掉 decode 后的 prefill-only cu_seqlens 生成,chunk_indices/chunk_offsets 也按 prefill-only 尾部准备,避免每层重复推导。输出按 decode-first 顺序用 `torch.cat` 缝合。附带改动:为 Qwen3.5-397B-A17B-NVFP4 增加 MTP eval 配置;`fast_kernel` 参数更名 `use_aiter`。

## 关键改动

```text
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: _forward_core 增加 split_non_spec 分支——decode 前缀走 fused_sigmoid_gating_delta_rule_update,prefill 尾走 chunk kernel,输出 cat 缝合
vllm/v1/attention/backends/gdn_attn.py: GDNAttentionMetadata 新增 prefill_query_start_loc/prefill_state_indices/prefill_has_initial_state,build 时按 decode 剥离 rebase
tests/kernels/mamba/test_gdn_forward_core_split.py: 新增 296 行集成测试(split 与 unified 在 core_attn_out 与 SSM state 上对齐)
```

关键代码摘录(节选,<=15 行):

```python
split_non_spec = (
    spec_sequence_masks is None
    and attn_metadata.num_prefills > 0
    and attn_metadata.num_decodes > 0
)
...
core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
    ..., initial_state=ssm_state, inplace_final_state=True,
    cu_seqlens=non_spec_query_start_loc[:num_decodes + 1],
    ssm_state_indices=non_spec_state_indices_tensor, ...)
...
core_attn_out_non_spec = torch.cat(
    [core_attn_out_decode, core_attn_out_non_spec], dim=1)
```

## 性能声明

- kernel 微基准(B200,cold-L2 CUPTI,397B-A17B-NVFP4,混合 953 decode + 7239 prefill):unified chunk 4361µs → split 2256.5µs,1.93x speedup(各配置 1.88-2.17x)。
- E2E(8xB200,DP=8+EP,Qwen3.5-397B-A17B-NVFP4,ISL 8192 / OSL 256,concurrency 2048,steady-state):Total tok/s 131,976 → 164,156,+24.4%。
- 准确性:GSM8K(1319 题,5-shot,B200):Qwen3.5-35B-A3B 0.8499(阈值 0.84)、Qwen3.5-35B-A3B-FP8 0.7953(阈值 0.79)、Qwen3.5-397B-A17B-NVFP4 0.8863(阈值 0.88),均通过。

## 验证与回归风险

- 新增 tests/kernels/mamba/test_gdn_forward_core_split.py:真实 `_forward_core` 与真实 metadata builder 下,split 路径与 unified chunk-only 路径在 core_attn_out 与 SSM state 上对齐(bf16/fp32 x 混合 batch 形状,8/8 passed,B200,SM10x 限定,强制 CuteDSL chunk backend);conv state 逐位相等,recurrent/chunk 累加顺序差异用容差(6e-2/2e-2)覆盖。
- 回归风险面:GDN(Qwen3.5 系)非 spec 混合 batch 的执行路径改变;decode 剥离依赖 batch 的 decode-first 布局与 metadata 的一致性;chunk 与 recurrent kernel 数值累积顺序不同,精度容差已校准;仅 CUDA SM10x(CuteDSL)路径被测试覆盖,FLA/Triton chunk backend 分支逻辑同改但未在本 PR 测试矩阵中。

## 关联

- 前序:PR #43273(CuteDSL GDN prefill backend,本 PR 的 chunk kernel 来源);MTP spec 路径已走同一 recurrent kernel。
- 同区域:索引库已有 pr-42311(flatten GDN prefill)、pr-59735(keep non-spec GDN decode on)、pr-61855(triton fused gdn)、pr-61867(specialized decode cutedsl)等 GDN 卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
