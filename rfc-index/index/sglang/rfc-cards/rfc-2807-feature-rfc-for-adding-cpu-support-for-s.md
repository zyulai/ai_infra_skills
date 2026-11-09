# RFC #2807: [Feature] RFC for adding CPU support for SGLang

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2807](https://github.com/sgl-project/sglang/issues/2807) |
| 作者 | mingfeima |
| 创建日期 | 2025-01-09 |
| 状态 | ⚪ closed (2025-07-21) |
| labels | cpu, enhancement, high priority, intel |
| 热度 | 👍 14 · 💬 14 |
| 分类 | Hardware / Backend |

## 一句话结论
Intel 团队为 SGLang 增加 CPU 设备支持,并针对带 AMX 的 Xeon 做 x86 推理性能优化。

## 提案动机
Intel PyTorch 团队(作者 Mingfei,CPU 性能模块维护者)希望向 SGLang 贡献 CPU 使能与性能优化。主要目标是 Intel Xeon Scalable(x86 服务器 CPU),聚焦支持 Intel AMX 的 Sapphire Rapids(4th gen)、Emerald Rapids(5th gen)、Granite Rapids(6th gen);对其他 ISA 的 CPU 提供原生实现或回退以保证可用,追求高性价比(good performance per dollar)。

## 提出的方案
- 先用 torch native backend 使能 CPU 设备,再逐步用 C++ intrinsics 内核替换性能关键组件(CPU 使能 PR #2806)。
- fused 内核:rms_norm、silu_and_mul、sampling 等。
- radix attention 的 extend 与 decode 内核。
- DeepSeek 性能优化(对齐 LMSYS 博客的 DeepSeek MLA 吞吐优化工作):MLA 解码 kernel 的 head blocking 优化、DeepSeekMoE(FusedMoE)、fp8 KV cache(实验性)。
- 张量并行:把 TP 映射到单节点 CPU 的多 socket(NUMA 节点);EPMoE 在计划中。
- 限制:kernel 使用 avx512 与 amx-bf16 编写,需 GCC11 及以上;BFloat16/Float16 在 CPU 上同时启用,但当前阶段只做 BFloat16 性能优化,Float16 后续再补。

## 优化点与收益
正文未给出量化数字;定性目标为「good performance per dollar」。

## 现状与落地
已关闭(2025-07-21)。25Q1 计划中除 EPMoE 外的条目均已勾选完成;正文附有 CPU 使能 PR #2806,其余各项落地情况需查关联 PR。

## 关联
- CPU 使能 PR #2806: https://github.com/sgl-project/sglang/pull/2806
- Intel AMX 概览: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html
- DeepSeek MLA 吞吐优化博客: https://lmsys.org/blog/2024-09-04-sglang-v0-3/

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
