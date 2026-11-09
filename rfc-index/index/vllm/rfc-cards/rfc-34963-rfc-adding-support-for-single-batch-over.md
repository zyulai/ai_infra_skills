# RFC #34963: [RFC]: Adding Support for Single Batch Overlap (SBO) With FlashInfer DeepEP LL NVFP4

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34963](https://github.com/vllm-project/vllm/issues/34963) |
| 作者 | elvircrn |
| 创建日期 | 2026-02-20 |
| 状态 | ⚪ closed(关闭于 2026-06-27) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |

## 一句话结论
提议为 vLLM 引入 Single Batch Overlap(SBO),把 DeepEP 的 combine all-to-all 与 GEMM2 重叠(SGLang 已实现,vLLM 尚无)。

## 提案动机
DeepEP 的 combine_v2 kernel 通过轮询 per-expert 完成信号,把 combine all-to-all 与 GEMM2 重叠,而不是等所有 expert 完成后再发送。SGLang 已实现该能力,vLLM 没有。作者在 E8/E32 P/D、500 ISL 1500 OSL、DeepSeek R1 NVFP4 的设置下做 trace:decode 中 combine 步骤耗时显著高于 GEMM op;启用 EPLB 后显示出可能更大的延迟降低空间。

## 提出的方案
- 参考 SGLang 实现:python/sglang/srt/batch_overlap/single_batch_overlap.py(CombineOverlapArgs / DownGemmOverlapArgs 数据类、信号张量创建、SM 分区);flashinfer_cutedsl_moe.py(向 GEMM2 传 sm_count=down_sm_count、dst_signals=down_signals);deepep.py(在辅助 stream 上以 overlap=True 启动 combine,带 src_signals / src_signal_expect_value)
- 已有实现尝试:mzx29z 已在 minosfuture/vllm 分支 pd_gb200_0114 上实现 SBO,需修改 FlashInfer(flashinfer sbo_extend_local_expert_0109 分支)与 DeepEP(mxz297/DeepEP sbo 分支)
- DeepEP 侧:上游 fzyzcjy/DeepEP 的 gb200_blog_part_2 分支提供 combine_v2(见 lmsys GB200 博客第二篇);elvircrn/DeepEP 的 gb200_blog 分支含 CUDA 13 补丁版 setup.py,使用更方便

## 优化点与收益
正文未给出量化数字,仅以 trace 图定性说明:combine 耗时显著长于 GEMM、存在重叠空间(E8/E32 P/D、500 ISL 1500 OSL、DeepSeek R1 NVFP4 的 trace 条件)。

## 现状与落地
已关闭(2026-06-27),labels 含 stale;正文未说明是否实现,落地情况需查关联 PR。

## 关联
- https://lmsys.org/blog/2025-09-25-gb200-part-2/
- https://github.com/fzyzcjy/DeepEP/tree/gb200_blog_part_2/
- https://github.com/minosfuture/vllm/tree/pd_gb200_0114

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
