# RFC #36459: [RFC]: vLLM IR Out-of-Tree (OOT) Kernel Registration

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36459](https://github.com/vllm-project/vllm/issues/36459) |
| 作者 | wxsIcey |
| 创建日期 | 2026-03-09 |
| 状态 | ⚪ closed(关闭于 2026-04-09) |
| labels | RFC, vllm-ir |
| 热度 | 👍 3 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
把内核模块加载权交给 `Platform.import_kernels()`,让 OOT 平台不加载 vLLM 内置 C 扩展即可注册自己的 IR 内核实现。

## 提案动机
#32358 提出 vLLM IR,PR #33825 建立了 vLLM IR 基础设施:用 `IrOp`/`IrOpImpl` 做 op 注册与基于优先级的分发,用 `VllmIrLoweringPass` 在 Inductor 编译期把 IR op lower 到具体实现;后端内核模块(`vllm_c.py`、`aiter_ops.py`、`oink_ops.py`)通过 `@rms_norm.register_impl(...)` 注册实现。但原来的 `vllm/kernels/__init__.py` 无条件导入全部内核模块,这对 OOT 平台不友好——OOT 硬件后端可能没有 CUDA/ROCm C 扩展可用,且需要在不加载 vLLM 内置内核的情况下注册自己的 IR 实现。

## 提出的方案
- 核心思想:内核注册入口改为 `Platform.import_kernels()`。`vllm/kernels/__init__.py` 不再直接导入内核,而是先 `import vllm.ir.ops`(保证 `rms_norm` 等 IrOp 对象先于任何 `register_impl` 调用创建,这是关键的顺序保证),再调用 `current_platform.import_kernels()`。
- 基类 `Platform.import_kernels()` 为 no-op;`CudaPlatformBase` 负责加载 `vllm._C`、`vllm._moe_C`、`vllm.kernels.vllm_c`、`vllm.kernels.oink_ops`(oink 对应 SM100 Blackwell);`RocmPlatform` 加载 `vllm._rocm_C`、`vllm_c` 与 ROCm 专属的 `aiter_ops`。
- OOT 平台只需在 `Platform` 子类中覆写 `import_kernels()`,导入自己的内核注册模块,用 `@ir.ops.rms_norm.register_impl("my_platform", supported=True, supports_args=...)` 注册实现,无需加载任何 vLLM C 扩展。
- 优先级设置:注册与优先级激活在 `IrOpPriorityConfig.set_priority()` 内原子完成,不存在"实现已注册但尚未被分发"的窗口;默认优先级由 `current_platform.get_default_ir_op_priority()` 提供,用户可通过 `KernelConfig.ir_op_priority` 前置自定义 provider,启动时由 `KernelConfig.set_platform_defaults()` 合并。
- OOT 平台只需实现 `get_default_ir_op_priority()`,即可声明自己偏好的分发顺序(如 `["my_platform", "native"]`)。

## 优化点与收益
正文未给出量化数字。收益是接入性而非性能:OOT 硬件后端不再被强制加载 CUDA/ROCm C 扩展,即可进入 vLLM IR 的注册与优先级分发体系。

## 现状与落地
已关闭(关闭于 2026-04-09)。正文的 Any Other Things 为 None,未给出实现 PR 或讨论结论,落地情况需查关联 PR。

## 关联
- [#32358](https://github.com/vllm-project/vllm/issues/32358)(vLLM IR 提案)
- [#33825](https://github.com/vllm-project/vllm/pull/33825)(vLLM IR 基础设施 PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
