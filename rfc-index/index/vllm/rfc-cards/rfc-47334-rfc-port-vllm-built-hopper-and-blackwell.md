# RFC #47334: [RFC]: Port vLLM-built Hopper and Blackwell C++ CUDA kernels to CuTeDSL

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47334](https://github.com/vllm-project/vllm/issues/47334) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-07-01 |
| 状态 | 🟢 open |
| labels | RFC, unstale |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Attention / Kernel |

## 一句话结论
提议把 vLLM 构建的 Hopper/Blackwell CUDA C++ kernel 尽量移植到 CuTeDSL,实现免 CUDA 构建的开发安装。

## 提案动机
vLLM 源码构建成本高昂,主因是自定义 CUDA/C++ kernel,拖慢本地搭建、日常开发、kernel 实验与 CI/调试周期。随着 vLLM 不断加入 SM90、SM100、SM103、SM120、SM121 专属 kernel,Hopper/Blackwell 上该问题持续加剧。CuTeDSL 提供了一条保留架构特化优化 kernel、同时减少 vLLM 直接构建的 CUDA/C++ 代码量的路径。目标延续 #17419(Kernel Library Restructure / Packaging Split)的动机:让贡献者在常见 Hopper/Blackwell 环境下安装 vLLM 并迭代 kernel,无需本地构建 vLLM CUDA C++。

## 提出的方案
- 将 vLLM 构建的所有 Hopper 与 Blackwell CUDA kernel 在技术可行的前提下移植到 CuTeDSL;终态是 Hopper/Blackwell 开发安装不再需要本地 CUDA C++ 构建,且裸 CUDA 性能保持有竞争力。
- 主要移植对象:量化 GEMM、MoE kernel、sparse/compressor kernel,以及 attention 邻近 kernel(CuTeDSL 或 FA4 是其长期路线)。
- Hopper attention 方向:待 FA4 decode 性能优化充分后以 FA4 取代 FA3,CuTeDSL 工作应面向前瞻性的 FA4 路线,不在 FA3 上重复长期投入。
- 不移植 Ampere kernel;不做无性能验证的重写;不投入预计会被 FA4 取代的 FA3 专属 CuTeDSL 工作。
- 服务期 JIT 编译及其 ITL 尖峰所需的 warmup/cache/JIT-monitoring 基础设施,留待后续 RFC 讨论。
- 每个 port 需报告:裸 CUDA kernel 时间与端到端服务影响、冷 JIT 与已预热行为,Hopper 与 Blackwell 结果分开报告。
- 正文列出已有的 CuTeDSL kernel 工作(PR #43584、#44230、#44501、#42899、#42562、#44343)与运维工作(#46167 运行时监控、#46182 warmup 基础设施、#45270 JIT warmup/cache PoC)。

## 优化点与收益
收益主要是构建/开发效率。正文给出构建计时数据(commit `40e5522121`(main,2026-06-23),8x H200 主机,CUDA 13.0,PyTorch 2.11.0+cu130,Ninja 1.13.0,`MAX_JOBS=160`):
- `uv pip install -e . --torch-backend=auto` 墙钟 **930s**;冷 cache(无 ccache/sccache)**747s**。
- Ninja 记录 392 个计时构建步骤、墙钟 727s;CUDA 目标编译占 369 步、并行累计编译时间 **55,383s**。
- 分项:vllm-flash-attn FA3 194 步 / 32,740s(最慢单文件 299s);FA2 55 步 / 11,736s;核心 `_C_stable_libtorch` 78 步 / 7,466s(最慢 `scaled_mm_c2x.cu` 724s);MoE `_moe_C_stable_libtorch` 26 步 / 2,497s;FlashMLA 27 步 / 1,321s。
成功标准:常见 Hopper/Blackwell 开发安装免本地 CUDA C++ 构建、wheel 构建时间显著下降、裸性能与现有 native kernel 相当、端到端服务不回退、服务期不发生运行时 JIT。

## 现状与落地
仍开放(labels: RFC, unstale)。正文为提案与数据论证阶段,明确后续还有 warmup/缓存 RFC;已有多项 CuTeDSL kernel/运维 PR 在推进中,整体移植尚待开展。

## 关联
- [#17419 Kernel Library Restructure / Packaging Split](https://github.com/vllm-project/vllm/issues/17419)
- [PR #43584](https://github.com/vllm-project/vllm/pull/43584) sparse compressor、[PR #44501](https://github.com/vllm-project/vllm/pull/44501) FP8/INT8 scaled-mm、[PR #42899](https://github.com/vllm-project/vllm/pull/42899) DSV4 indexer(已有 CuTeDSL kernel 工作举例)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
