# PR #46546: [ROCm][ [Perf] sparse attention optimization on minimax-m3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hongxiayang |
| 合并 | 2026-06-25 |
| 分类 | Attention / Kernel |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/46546 |

## 一句话结论

把 MiniMax-M3 的 lightning-indexer 打分/top-k 与 block-sparse attention Triton 内核从 gfx942 扩展到 gfx950,TP8 下 TTFT 中位数减半(2.02x)。

## 动机

前序 PR #46035 只针对 gfx942 做 sparse attention 优化;gfx950(MFMA 形状、pipeline 参数不同)无对应实现。MiniMax-M3 采用 lightning-indexer 结构:先对每个 128-token KV block 打分(idx_q 与 index-K 的点积取 block 内 max),选 top-k block,再让主注意力只 attend 被选中 block。缺 gfx950 路径时该模型在新平台上跑不快。

## 优化点

新增 `amd/ops/index_topk.py`:prefill 路径用 `_index_block_score_kernel`(paged,每 K-tile 恰好一页)加 bitonic 归并的 `_topk_index_kernel`;decode 路径做 split-K——`_decode_index_score_kernel` 按 KV chunk 并行打分(grid 只依赖 shape 常量,cudagraph 安全),`_topk_index_partial_kernel` 各 chunk 局部 top-k,`_topk_index_merge_kernel` 归并全局 top-k,init/local block 用 1e30/1e29 分数强制入选。index-K cache 支持 BF16 与 FP8(e4m3)存储,MMA 以 fp32 累积保证 block max 精确;支持 PDL(gdc_wait/launch_dependents,ROCm 上恒关)。`amd/ops/sparse_attn.py` 为 prefill 专门调 CDNA 子分块:gfx950 每 block 拆 64、gfx942 拆 32(`SPARSE_BLOCK_SIZE//2` vs `//4`),MFMA 参数 `num_warps=1, matrix_instr_nonkdim=16, kpack=2, num_stages=1`。KV page size 强制等于 sparse block size 128(一 block 一页)。

## 关键改动

```text
vllm/models/minimax_m3/amd/ops/index_topk.py: 新增打分 + bitonic top-k + decode split-K 三组 Triton 内核
vllm/models/minimax_m3/amd/ops/sparse_attn.py: gfx950/gfx942 分架构的 block-sparse GQA prefill 内核
```

## 性能声明

minimax-m3-mxfp8,gfx950。TP8(8192-in/1024-out,random 数据集,并发 64):output throughput 1183.8 → 1683.3 tok/s(+42.2%),median TTFT 1896 → 937 ms(-50.6%,2.02x),median TPOT 51.4 → 36.2 ms(-29.5%);并发 1:median TTFT 496 → 253 ms(-49%,约 2x)。另一组(8k/1k,并发 64,未注明 TP):output throughput 1269.98 → 1351.42 tok/s(+6.4%),median TTFT 1372.6 → 1228.0 ms(-10.5%),median TPOT -6.0%;并发 1 median TTFT -10.9%。

## 验证与回归风险

正确性:GSM8K 全量 1319 题、5-shot、temp 0,flexible-extract 0.9143 → 0.9136、strict-match 0.9136 → 0.9128(均在 stderr 噪声内)。风险面:仅 ROCm gfx942/gfx950 生效,decode/`spec decode` 的多 head 宽 query tile 路径单独降 num_warps/num_stages;page size 与 sparse block size 强耦合(128);Triton autotune 配置集对新 arch 需重调(`tune_sparse_attn.py`)。

## 关联

同区域:PR #46035(gfx942 前序,本 PR 是其 gfx950 扩展);MiniMax-M3 sparse MLA / lightning indexer 相关内核 PR;Co-Authored-By @yueliu14。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
