# PR #53166: [Perf][ROCm][Attention] Fuse Kimi-K3 MLA chunked-context gather on AITER

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nsusanto |
| 合并 | 2026-10-27 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, kimi, k3 |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/53166 |

## 一句话结论

ROCm MLA chunked prefill 上下文准备从 4 个 kernel(gather、GEMM、两次 copy)融合为单个 AITER `gather_kv_b_proj` launch。

## 动机

dense-MLA prefill 的 `_compute_prefill_context` 对每个 context chunk 要发 4 个 kernel:`gather_and_maybe_dequant_cache`、`kv_b_proj` 展开 GEMM(Cijk_…MT192x160x64)、以及把 k_nope 与 k_pe 拼进 k 的两次 `direct_copy`。AITER 已自带 `gather_kv_b_proj`,一次 launch 即可完成全部四步,但 vLLM 尚未接入。

## 优化点

机制:`AiterMLAImpl._compute_prefill_context` 每个 chunk 改调 AITER `gather_kv_b_proj`,直接从 FP8/BF16 cache gather 并展开为 (k, v);每步的 per-token KV 索引由 metadata builder 一次性构建(`_build_context_chunk_kv_indices`,把 chunk 的 block table 用扩展了 `HAS_START_OFFSETS` 的 `_expand_page_indices_kernel` 摊平),避免每层 MLA 重复展开。融合仅在 `dcp_world_size == 1`、gfx950、非 `fp8_ds_mla` cache、未量化 kv_b_proj 等条件下启用(`_can_fuse_context_gather`),DCP>1 保留 main 的 DCP prefill 路径。另外 `_concat_k_nope_k_pe` 改用已合并的 `fused_kimi_k3_mla_kv_concat`(#51772)替代两次 slice-assign copy。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: 融合 gather 路径 + per-chunk KV 索引构建 + 融合 k concat
tests/kernels/attention/test_rocm_aiter_mla_prefill_fusions.py: 新文件,11 个用例
```

## 性能声明

Kimi-K3 MXFP4 agentic recipe,8x MI355X,AITER v0.1.23,`vllm bench serve` random 数据集(warm-up 后单次运行):
- Conc 14(DCP 1,融合开):1k/1k output throughput 681.60→692.60 tok/s (+1.6%),TTFT 342.00→339.00 ms (−0.9%),TPOT 19.46→19.23 ms (−1.2%);8k/1k throughput 292.10→299.04 tok/s (+2.4%),TTFT −0.5%,TPOT 44.94→43.81 ms (−2.5%)。
- Conc 70 DCP 1:8k/1k throughput 1103.27→1118.97 tok/s (+1.4%),TTFT 3167.12→3103.83 ms (−2.0%),TPOT −1.4%。
- Conc 70 DCP 8(生产 recipe,融合被关闭):各指标 ±0.4% 以内。GSM8K(5-shot)0.9591。

## 验证与回归风险

新增 `tests/kernels/attention/test_rocm_aiter_mla_prefill_fusions.py`:融合 vs 未融合路径对齐(atol=rtol=2e-2)、KV 索引与 block table 逐 chunk 一致、5 类不支持输入正确回退、k concat 与 `torch.cat` 相等;并用计数器确保融合真的发生(防止静默回退假通过)。风险面:仅 ROCm AITER MLA prefill;`fp8_ds_mla` cache、FP8 query、量化 kv_b_proj、非连续 cache 等自动回退;DCP>1 不受影响。MI355 上 k/v 输出与 torch GEMM 可差 1 bf16 ulp(实测 2e-3)。

## 关联

同区域:#51772(`fused_kimi_k3_mla_kv_concat` 的来源)、#59070(后续把 AITER gather 路径扩展到 DCP>1 的 FP8 AllGather 融合)、#59068 系列 Kimi-K3 ROCm 优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
