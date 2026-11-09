# RFC #33214: [RFC]: XPU kernel migration to vllm-xpu-kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33214](https://github.com/vllm-project/vllm/issues/33214) |
| 作者 | jikunshang |
| 创建日期 | 2026-01-28 |
| 状态 | ⚪ closed (关闭于 2026-06-22) |
| labels | RFC |
| 热度 | 👍 22 · 💬 1 |
| 分类 | Attention / Kernel |

## 一句话结论
把 vLLM 的 Intel GPU(XPU)内核支持从 IPEX 迁移到专用库 vllm-xpu-kernels,提升性能、可维护性与集成质量。

## 提案动机
vLLM 的 Intel GPU 支持此前依赖 Intel Extension for PyTorch(IPEX),其存在明显局限:依赖沉重;IPEX/PyTorch/oneAPI 之间存在版本兼容问题;发布周期受限。专用库 vllm-xpu-kernels 迭代快、可维护性更好、轻量,且专为 vLLM 推理而构建。

## 提出的方案
- 迁移周期 2 周:在 v0.15.0 发布后开始,0.16.0 发布前完成;拆分为多个 PR 以平滑快速推进
- 0.16.0 里程碑(全部标注 Done):
  - vllm-xpu-kernels 发布 v0.1.0
  - 移除 IPEX 依赖,用 vllm-xpu-kernels 替换 rms_norm/rope/activation/attention 内核(PR #33379)
  - int4 compressed_tensor_w4a16(#33973)
  - fp8 gemm/model 支持(w8a16、w8a8)(#34117)
  - 未量化 MoE(#33659)、fp8 MoE(#34202)、mxfp4 MoE(#33679)支持
  - IPEX 清理与文档更新(#34111)
- 0.16+ 里程碑:int4 gemm/model 内核(awq/gptq,WIP #33662)、int4 MoE、lora fix(WIP #34115)、inc 及 TBD 项
- 只改内核部分,torch.compile 等其他特性不受影响

## 优化点与收益
正文未给出量化数字。定性收益:去掉沉重依赖与版本兼容问题(IPEX/PyTorch/oneAPI),改用迭代快、轻量、为 vLLM 推理专用的内核库,改进 Intel GPU 后端的性能、可维护性与集成质量。

## 现状与落地
已关闭。正文清单显示 0.16.0 里程碑各项均标注 Done 并附 PR 链接;0.16+ 里程碑中 int4 gemm(awq/gptq)与 lora fix 标注 WIP,int4 MoE、inc、TBD 未完成,需后续跟进。

## 关联
- vllm-xpu-kernels:https://github.com/vllm-project/vllm-xpu-kernels
- PR #33379
- PR #34117
- PR #33659
- PR #33662

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
