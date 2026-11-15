# PR #56451: [ROCm][Spec Decode] Enable adaptive DSpark verification for MiniMax M3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @olka-amd |
| 合并 | 2026-11-11 |
| 分类 | Speculative Decoding(tier: major) |
| Label | rocm, speculative-decoding, ready, qwen, nvidia, mrv2, dflash, minimax |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/56451 |

## 一句话结论

给 MiniMax M3 的 ROCm Triton indexer/sparse-decode 路径加上 ragged decode 支持,从而在 ROCm 上启用 adaptive DSpark verification:Sonnet 负载 MXFP4 TP=2 输出 tok/s +3.8%~+5.9%,并解决固定 verify 在高并发下的 OOM 与 GPU Hang。

## 动机

Adaptive DSpark backend 扩展(#51303)的一部分:每个模型需要 varlen-capable sparse kernels,adaptive verification 才能把 draft tokens 在请求间重分配。Scope 收得很窄:只有 ROCm Triton indexer 与 Triton sparse-decode 路径获得 ragged decode 支持(在 MI325/gfx942 与 MI355X/gfx950 验证);CUDA backends、AITER MSA indexer / AITER sparse PA、ROCm NVFP4 sparse path、indexer CP path 继续上报 device/CPU query-length mismatch 不支持,selector 在这些路径保持 adaptive verification off——关闭时 uniform decode path 完全不变。

## 优化点

1. MiniMax M3 indexer 与 sparse attention(ROCm Triton)改为 device-side ragged decode metadata:两种 AMD index-score launch policy、fused top-k selector 与 sparse decode 都从 device `query_start_loc` 把 packed query tokens 映射回请求,使用 upstream 的 `compute_token_to_req_indices`(device-side 二分,不依赖 `max_query_len`,处理 zero-length padding rows);共享 builder 逻辑放 `models/minimax_m3/common/varlen_decode.py`。
2. Cudagraphs:builders 保持 `UNIFORM_BATCH`,仅在 adaptive verification 启用时声明 bounded varlen contract(`get_varlen_cudagraph_max_query_len` = 1 + num_speculative_tokens);mixed prefill+decode batch 永不 replay FULL graphs。
3. `supports_device_cpu_query_lens_mismatch()` 仅在 base rule 成立且 builder 拥有 ragged decode kernels 时为 true。
4. `kernel_paged_attention_2d` 在 `query_len != 1` 时 return(原为 `> 1`):FULL cudagraph replay 用 zero-length rows pad request 维度,这些行此前落进 decode path 并读写越界一行;`ROCM_ATTN` 报 `ALWAYS`,故可 replay adaptive verification 产生的 ragged decode graphs。
5. Indexer CP(`minimax_m3_indexer_cp`)显式拒绝 adaptive verification(其 decode 只打分 uniform rows)。
6. 顺带移除了早前放在此处的 DSpark confidence-head config fallback。

## 关键改动

```text
vllm/models/minimax_m3/common/varlen_decode.py(新增,diff 未展示): VarlenDecodeBuilderMixin + build_decode_query_layout + make_token_to_req_buffer
vllm/models/minimax_m3/amd/ops/index_topk.py: index-score / fused top-k Triton kernel 增加 query_start_loc、token_to_req 与 VARLEN constexpr
vllm/models/minimax_m3/common/ops/sparse_attn.py: _gqa_sparse_decode_kernel 同样接入 VARLEN
vllm/models/minimax_m3/common/indexer.py: builder 混入 VarlenDecodeBuilderMixin;supports_device_cpu_query_lens_mismatch 门控;CP 拒绝 adaptive
vllm/models/minimax_m3/amd/sparse_attention_nvfp4.py: supports_varlen_decode = False(NVFP4 decode kernel 只打分 uniform rows)
tests/kernels/attention/test_minimax_m3*.py / test_prefix_prefill.py: ragged + padded rows 测试
```

(注:pack 的 diff 截断,有 4 个文件未展示,以上仅覆盖已展示部分。)

```python
    if VARLEN:
        q_start = tl.load(query_start_loc_ptr + pid_r)
        q_len = tl.load(query_start_loc_ptr + pid_r + 1) - q_start
    else:
        q_start = pid_r * decode_query_len
        q_len = decode_query_len
    q_mask = q_offsets < q_len
    q_ids = q_start + q_offsets
```

## 性能声明

E2E 基线:MI325,`nvidia/MiniMax-M3-DSpark` drafter,`num_speculative_tokens=7`,FULL cudagraphs,默认内存配置,`--max-model-len 4096 --max-num-seqs 512`,同一 build 上 fixed vs adaptive 对比;drafter 声明 confidence head 但不带 `confidence_head.*` 权重,adaptive 使用 online acceptance estimator(#52228)。

- **GSM8K 全量**(1,319 questions,5-shot,greedy,concurrency 64):MXFP4 TP=2 0.8196→0.8226;MXFP8 TP=4 0.8779→0.8711;BF16 TP=8 0.8704→0.8688。MXFP8 重复 run 两种模式各约 ±0.01 波动(fixed 0.860–0.878 / 6 runs,adaptive 0.867–0.879 / 9 runs)。
- **MI355X(gfx950)**,MXFP4 TP=2,`--max-model-len 32768`:GPQA diamond(0-shot CoT,flexible-extract)0.7929→0.7929;NIAH single 1/2/3 @4K/8K/16K(9 次均值)0.987→0.986。
- **Sonnet 550 in / 150 out / 200 shared prefix, ignore EOS, output tok/s(adaptive vs fixed)**:MXFP4 TP=2:C128 649.96→674.59(+3.8%)、C256 +5.4%、C320 +5.7%、C380 +5.7%、C420 +5.9%、C500 +5.3%;MXFP8 TP=4:C128 −3.3%、C256 +1.8%、C320/C380 +3.4%、C420 −0.8%、C500 fixed OOM→adaptive 1929.71;BF16 TP=8:C128 −3.6%、C256 +1.5%、C320 +4.1%。正文解释:TP=4/TP=8 下 verify 成本低,adaptive 可省的余地小;MXFP8 fixed 在 C500 分配完整 verification logits(1.6 GB)OOM,adaptive 限制该分配并跑完全部请求。
- **BF16 TP=8**(#62769 之后最新 build,c512,1k OSL):adaptive 3,322.8 tok/s(acceptance length 2.28)vs fixed k=7 3,082.1 tok/s(acceptance length 3.3);GSM8K 0.961 vs 0.959;此前 C≥380 会 `GPU Hang` 的配置现在通过。
- **Rebase 前早期结果**(MXFP4 target,`olka-fi/MiniMax-M3-MXFP4-DSpark` drafter,TP=2,同 Sonnet 负载,3–5 分钟稳态,9,040 requests 无失败):增益 C128 +10.1% 至 C500 +28.6%,median TPOT C500 667.91→521.15 ms。

## 验证与回归风险

- `tests/kernels/attention/test_minimax_m3.py`(含 ragged index top-k、带 zero-length padding rows 的 ragged sparse decode,TP1–TP8 head shapes):MI325(gfx942)与 MI355X(gfx950)通过;ragged index top-k 覆盖 gfx950 全部 indexer decode launch path(balanced、high-batch、default score launches;single- 与 16-chunk topk=16 selector)。
- `test_minimax_m3_varlen_cudagraph.py`(新增)、`test_minimax_m3_indexer_cp.py` 通过;`test_prefix_prefill.py::test_contexted_kv_attention`(增加 trailing zero-length row)gfx942/gfx950 通过;`tests/v1/spec_decode/test_adaptive_verification.py`、`test_gpu_rejection_sampler_chunking.py`、`test_attn_utils.py` 通过。
- 压力测试:serving 规模 batch 的 ragged index top-k 与 sparse decode kernel,60 个随机 batch(380–512 请求、每请求 1–8 query tokens、至 4K context、zero-length padding rows),每第 10 个 batch 对照 reference,全部通过。
- 回归风险:仅 ROCm Triton 路径启用;其余 backend adaptive 保持 off;`kernel_paged_attention_2d` 行为变化(`query_len != 1` 即 return)修复 FULL graph padding 下的越界读写;indexer CP 在 adaptive verification 开启时直接 raise ValueError。

## 关联

- #51303(adaptive DSpark backend 扩展)、#52228(online acceptance estimator)、#62769(BF16 最新结果所用 build)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
