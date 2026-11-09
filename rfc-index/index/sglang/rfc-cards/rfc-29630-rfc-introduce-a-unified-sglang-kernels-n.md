# RFC #29630: [RFC] Introduce a unified sglang.kernels namespace for kernel organization and dispatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29630](https://github.com/sgl-project/sglang/issues/29630) |
| 作者 | BBuf |
| 创建日期 | 2026-06-29 |
| 状态 | ⚪ closed (2026-10-01) |
| labels | RFC, inactive, jit-kernel, kernel |
| 热度 | 👍 13 · 💬 15 |
| 分类 | Attention / Kernel |

## 一句话结论
提议以 `sglang.kernels.ops.*` 作为内核统一公共导入面,归并散落各处的 Triton/JIT/AOT 内核;该迁移最终已全部完成。

## 提案动机
内核代码散布在 `python/sglang/jit_kernel`、`sgl-kernel/python/sgl_kernel`、`srt/**/triton_ops` 及模型专属路径,带来实际问题:难以判断某逻辑算子是否已有 kernel;同一算子的备选实现不易对比;相关内核测试的存放与导入方式不一致;JIT CUDA / AOT CUDA/C++ / Triton / CuTe DSL / FlashInfer / DeepGEMM / torch fallback 的后端覆盖难以审查;新内核 PR 缺少明确的公共 API 目标。

## 提出的方案
- 公共命名空间 `sglang.kernels.ops.<group>.<kernel>`,按 activation/attention/moe/quantization/kvcache/layernorm 等算子组组织
- 保守兼容:保持 `sgl_kernel` wheel 公共 API 稳定,`jit_kernel` 先作过渡层并以薄 wrapper 转发,风险路径先加 wrapper、后移实现
- 轻量元数据 `KernelSpec` / `KernelBackend` / `CapabilityRequirement` + 启发式 selector(按显式后端请求、平台能力、算子属性、优先级排序);自动调优暂缓
- 五阶段迁移:先强化内核正确性测试 → 建命名空间并迁移易迁移面(含 `srt/**/triton_ops`)→ Phase 2.5 全树清扫散落内核 → 共享 JIT 基础设施迁入 `kernels/jit` → 逐算子组删除转发
- Review 规则:新增可调用内核的 PR 必须提供 `sglang.kernels.ops.*` 入口,避免 `jit_kernel` 长期作为公共算子命名空间

## 优化点与收益
正文未给出性能量化数字;收益为工程性:Phase 2.5 清扫约 280 个散落的 Triton / CuTe DSL / TileLang 内核(约 105 个文件,拆 7 个 PR,#30784–#30795)。迁移完成后 `python/sglang/jit_kernel/` 已整体删除(#32072),`sglang.kernels` 成为 SGLang 内核的唯一归宿。

## 现状与落地
已关闭(2026-10-01)。正文头部标注「Status — migration complete ✅ (2026-07)」:全部阶段已落地、jit_kernel 已删除,并附各阶段与 PR 的对照表。

## 关联
- #30044(Phase 2,命名空间 + triton_ops 迁移)
- #31292 / #31307(metadata 与 HIP/NPU 后端)
- #31666(共享 JIT 基础设施)
- #32045、#32072(收尾,删除 sglang.jit_kernel)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
