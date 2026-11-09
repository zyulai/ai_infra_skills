# PR #22850: [AMD] Reduce NSA indexer kernels (weights_proj,  k-cache store kernel fusion)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-04-19 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat（diff 内容可用：1 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/22850 |

## 一句话结论

HIP 上 NSA indexer 消除冗余 kernel：weights_proj 3 -> 1 kernel（每层省约 10us）、k-cache store 2 -> 1 kernel（每层省约 4us），MI355X TP8 吞吐 +2.29%。

## 动机

NSA indexer 在 HIP 上有两处冗余 kernel。weights_proj：`ReplicatedLinear` 用 fp32 params_dtype（仅 CUDA 路径用 bf16），使 tgemm 无法派发到 tuned bf16 fused kernel 而回退 torch GEMM，且 GEMM 前还要一次 `x.to(fp32)` 转换——合计 3 个 kernel（dtype cast + fp32 GEMM + reduction）。indexer k-cache store：key 量化（act_quant）与缓存写入（set_index_k_scale_buffer）是两次独立 kernel launch。

## 优化点

1) weights_proj 的 `params_dtype` 统一为 bf16（与 CUDA 路径对齐），删除多余的输入 dtype cast；GEMM 单 kernel 派发到 aiter 的 `wv_splitk_small_fp16_bf16_kernel`；HIP 分支直接返回 bf16 权重（后续与 q_scale 相乘时会提升回 fp32），非 HIP 路径保留 `.float()`。2) `_store_index_k_cache` 增加 AITER 快速路径：page_size=1 时把 (num_pages,132) uint8 buffer reshape 为 kernel 期望的 (num_blocks, block_size, head_dim+scale_bytes) fp8 布局，单次调用 `indexer_k_quant_and_cache` 完成 quant + cache store，原 act_quant + set 两步作为 fallback 保留。另清理了重复的 `_use_aiter` 定义。

## 关键改动

```text
python/sglang/srt/layers/attention/nsa/nsa_indexer.py: weights_proj params_dtype 改 bf16；_store_index_k_cache 加 AITER indexer_k_quant_and_cache 快速路径
```

## 性能声明

每层 kernel 时间：weights_proj 省 ~10us（3 -> 1 kernel），k-cache store 省 ~4us（2 -> 1 kernel）。MI355X TP8，GLM-5.1-FP8，concurrency 4/8/16/32/64 平均（baseline: rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412 docker）：ISL/OSL 1k/1k Throughput +2.29%、TPOT -1.86%；ISL/OSL 8k/1k Throughput +0.14%、TPOT -1.05%。准确度：TP8 GSM8K，MI300（GLM-5-fp8）0.950，MI355（GLM-5.1-fp8）0.951。

## 验证与回归风险

无新增单测（Checklist 未勾选），以 GSM8K 覆盖（两代硬件均在 0.95 水平）。风险：weights_proj 输出 dtype 从 fp32 变为 bf16（HIP 路径），依赖与 q_scale 相乘的 fp32 提升，其他 HIP 消费点若假设 fp32 输出会有精度差异（PR 未逐一列出）；AITER 快速路径仅限 page_size=1、且要求 `out_loc` 连续（不连续时会补一次 `.contiguous()`）；非 aiter 环境、NPU/CUDA 路径走原逻辑不变。

## 关联

PR 正文未提及关联 PR；同区域为 NSA indexer / MLA 的 HIP 优化（如 SGLANG_ROCM_FUSED_DECODE_MLA 等 env 涉及路径），参见索引同分类条目。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
