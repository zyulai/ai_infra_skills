# RFC #18153: [RFC]: Blackwell Enablement for vLLM (SM100)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18153](https://github.com/vllm-project/vllm/issues/18153) |
| 作者 | pavanimajety |
| 创建日期 | 2025-05-14 |
| 状态 | ⚪ closed (关闭于 2026-02-16) |
| labels | RFC, nvidia, stale |
| 热度 | 👍 19 · 💬 10 |
| 分类 | Feature / Ecosystem |

## 一句话结论
作为 NVIDIA Blackwell(SM100)支持的跟踪清单,汇总 NVFP4、注意力 kernel、FP8 GEMM/MoE、通信 overlap 等各项使能工作。

## 提案动机
NVIDIA 正在为 vLLM 的 Blackwell 支持做增量改动,该 issue 作为所有计划项的 tracker(正文未展开更详细的动机论述)。

## 提出的方案
按功能块逐项推进(清单式跟踪):
- NVFP4 支持(已完成):线性层 NVFP4 kernels、NVFP4 MoE kernels、nvidia/*-FP4 模型集成(均为 NVIDIA 承担)、为 Blackwell 微调 GEMM 配置、MoE 延迟优化与吞吐优化(flashinfer PR !1113)、MoE All Reduce Fusion(flashinfer PR !1108)
- 通信 overlap 算子优化(部分完成):已启用 NCCL symmetric memory(#24532);Gemm + comm overlap 支持未完成
- Blackwell Attention Kernels(已完成):集成 Cutlass MLA Kernels(#17625)、集成 vLLM v1 兼容的 Blackwell prefill/decode GQA kernels(flashinfer PR !1051)
- FP8 Blockscale GEMM 与 MoE(已完成):FP8 Blockscale GEMM 及优化(#18564)、FP8 Blockscale MoE、延迟与吞吐优化
- MTP 支持(已完成)

## 优化点与收益
正文未给出量化数字。定性收益:NVFP4/FP8 量化路径与 Blackwell 原生 attention/MoE kernels 使 vLLM 可在 SM100 上高效运行;MoE 与通信 overlap 的优化面向延迟与吞吐。

## 现状与落地
已关闭。清单中除「Optimize communication overlap ops」下的「Add support for Gemm + comm overlap」一项未勾选外,其余均勾选完成;整体落地情况可查关联 PR。

## 关联
- flashinfer PR !1113:https://github.com/flashinfer-ai/flashinfer/pull/1113
- flashinfer PR !1108:https://github.com/flashinfer-ai/flashinfer/pull/1108
- flashinfer PR !1051:https://github.com/flashinfer-ai/flashinfer/pull/1051
- #17625
- #24532

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
