# RFC #18365: 🚀 [RFC] Decoupling Framework Logic from Hardware Kernels: Offloading Triton Kernels to sgl-kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18365](https://github.com/sgl-project/sglang/issues/18365) |
| 作者 | pre-master |
| 创建日期 | 2026-02-06 |
| 状态 | ⚪ closed (2026-02-08) |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |

## 一句话结论
提议把 layer/models/model_executor/speculative 内嵌的 Triton 内核迁出到 sgl-kernel,高层 Python 只经标准化 API 调用内核,实现「去 Triton 化」。

## 提案动机
Triton 深度依赖硬件特性(register pressure、shared memory size、L2 cache 策略等),与高层 Python 逻辑紧耦合,妨碍框架维护与跨平台兼容(如 NVIDIA CUDA 迁移到 AMD ROCm 或其他加速器)。痛点:改 `models` 高层逻辑被迫管理 `grid` / `num_warps` / `num_stages` 等底层参数;`model_executor` 目前处理 Triton 编译与环境检查(硬件锁定);相似内核逻辑因缺统一管理层在标准推理与投机解码间重复;驱动/Triton 版本变化易引发主框架回归。

## 提出的方案
- 四层迁移映射:Layer 定义抽象层、实际计算导出到 `sgl_kernel.ops`;Models 只保留参数加载与架构编排;Model Executor 专注任务调度、抽象掉 Triton 编译与环境检查;Speculative 抽出复杂采样与验证内核
- 在 sgl-kernel 建统一 Entry Point:接口只接收 Tensors 与 metadata,硬件调参(Auto-tuning)封装进内核库
- 调用链改造:`from sglang.srt.layers.kernels.attention import triton_attention_kernel` → `import sgl_kernel; sgl_kernel.ops.attention(...)`
- 环境隔离:主包去除对特定 Triton 内部版本的硬依赖,sgl-kernel 作中间层处理硬件兼容检查与运行时分发
- 三阶段路线:Phase 1 Layer & Models(目标「Pure Python」模型定义)→ Phase 2 Speculative(加速投机算法迭代)→ Phase 3 Model Executor(调度与执行完全解耦)
- 已知代价:跨仓贡献工作流(以 `pip install -e` 缓解)、内核 API 需保持向后兼容

## 优化点与收益
正文未给出量化数字;收益为可维护性与可移植性:显著减少 `models` / `layer` 模块 LoC、sgl-kernel 可独立演进性能调优与硬件移植而不影响框架稳定性、为多 AI 芯片后端支持提供清晰路径。

## 现状与落地
已关闭(2026-02-08,创建两天后)。正文自标 Status: Draft,未见实现细节;落地情况需查关联 PR。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
