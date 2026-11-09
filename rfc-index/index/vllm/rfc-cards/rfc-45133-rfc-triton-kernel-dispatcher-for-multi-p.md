# RFC #45133: [RFC]: Triton Kernel Dispatcher for Multi-Platform Support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45133](https://github.com/vllm-project/vllm/issues/45133) |
| 作者 | wangxiyuan |
| 创建日期 | 2026-06-10 |
| 状态 | ⚪ closed(关闭于 2026-09-30) |
| labels | RFC |
| 热度 | 👍 5 · 💬 3 |
| 分类 | Attention / Kernel |

## 一句话结论
提议新增 Triton Kernel Dispatcher 模块,用装饰器实现平台级 kernel 覆盖,替代 vllm-ascend 的 monkey-patching。

## 提案动机
vLLM 在 sampling、attention、normalization、mamba/copy、投机解码等大量场景使用 Triton kernel,但这些 kernel 专为 GPU 设计优化,其他硬件无法直接运行。以 Ascend NPU 为例:功能上存在 grid 映射限制(grid 不得超过 AI Core 数、2D grid 须匹配硬件拓扑)、数据类型限制(如 Vector ADD 不支持 int64)、32/512 字节内存对齐要求、`tl.dot` 的 M/N/K tile 约束、缺失 CUDA 线程/流/事件/同步 API;性能上还存在 grid 并行度过高、缺少 NPU 所需的 inner-loop tiling(XBLOCK/XBLOCK_SUB,192KB UB 容量场景)、无法配置 triton-ascend 的 multibuffer 等 IR 优化、partial mask 填充引发多余 Vector-MTE 同步等问题。现状是 vllm-ascend 已重写超过 55 个 Triton kernel,但靠运行时 monkey-patching 替换上游 kernel,使代码库更复杂难维护,还可能导致隐蔽 bug 与性能问题。

## 提出的方案
- 新增模块 `vllm/model_executor/triton_dispatcher.py`,提供两个装饰器:
  - `@pluggable_kernel`:标注 vLLM 核心代码中的默认(CUDA)Triton kernel,使其可被替换。
  - `@register_kernel(name)`:硬件插件中按唯一名称注册平台特定实现。
- 调度行为:调用 `kernel[grid](args)` 时,有已注册 override 则调用 `override(args, grid=grid, **kwargs)`,否则回退到默认 Triton kernel 的原生调用语法——CUDA 上无 override、默认 kernel 零开销运行;NPU 及其他平台由插件在启动时注册各自实现。
- `KernelDispatcher` 类包装默认 kernel:`__call__` 分发到 override 或默认实现;`__getitem__(grid)` 对 override 返回 `_KernelLauncher`(grid 作关键字参数传入),对默认 kernel 委托 `default_kernel[grid]` 保持原生 Triton 行为。
- 保留标准 Triton 调用语法,现有调用点零改动。

## 优化点与收益
收益为多平台可维护性:用正式的注册/分发机制取代 monkey-patching,消除由此带来的复杂性、隐蔽 bug 与性能隐患;CUDA 路径零开销,硬件插件接入只需注册自己的实现。正文引用现状数据:vllm-ascend 已重写超过 55 个 Triton kernel。除此之外正文未给出量化数字。

## 现状与落地
已关闭(2026-09-30)。正文附有参考实现链接(PR #43048),反馈期未填;dispatcher 是否已合入主线从正文无法看出,落地情况需查关联 PR。

## 关联
- [参考实现 PR #43048](https://github.com/vllm-project/vllm/pull/43048)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
