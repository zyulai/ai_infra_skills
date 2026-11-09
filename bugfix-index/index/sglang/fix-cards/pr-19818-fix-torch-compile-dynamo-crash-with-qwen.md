# PR #19818: Fix torch.compile/dynamo crash with Qwen3 QK-norm in piecewise CUDA graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19818](https://github.com/sgl-project/sglang/pull/19818) |
| 作者 | jasperjiaguo |
| 合并日期 | 2026-03-20 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 29 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
Qwen3 QK-norm 的 JIT 判断函数在 dynamo 追踪时遇到 `logging.getLogger` 与文件系统 I/O 抛 `Unsupported` 崩溃;提级 logger 并加 `assume_constant_result` 修复。

## 问题现象
使用 `--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler inductor` 启动 Qwen3 模型(使用 QK-norm)时,服务器在 CUDA graph warmup 阶段崩溃,报 `torch._dynamo.exc.Unsupported`。触发条件为 piecewise CUDA graph + inductor 编译器组合与 QK-norm 模型共同使用。

## 根因
PR 正文披露:`can_use_fused_inplace_qknorm()` 在函数体内调用 `logging.getLogger(__name__)`。torch.compile/dynamo 追踪该函数时无法处理 `logging.getLogger()` 调用(与 `LoggingLoggerVariable` 的字符串比较失败)。此外该函数还通过 `_jit_qknorm_module()` 执行文件系统 I/O(JIT 模块编译),dynamo 同样无法追踪。两类不可追踪操作叠加导致 warmup 时 dynamo 抛 `Unsupported` 使服务器崩溃。

## 修复方案
对 `python/sglang/jit_kernel/norm.py` 两处修改:
1. 把 `logger = logging.getLogger(__name__)` 提升到模块级,使 dynamo 追踪期间永远不会遇到该调用。
2. 给 `can_use_fused_inplace_qknorm` 加 `@torch.compiler.assume_constant_result` 装饰器,让 dynamo 以 eager Python 方式求值该函数并内联布尔结果,从而完全避免追踪进文件系统/JIT 操作。

关键 diff:

```diff
+logger = logging.getLogger(__name__)
+
 from sglang.jit_kernel.utils import (
...
+@torch.compiler.assume_constant_result
 @cache_once
 def can_use_fused_inplace_qknorm(head_dim: int, dtype: torch.dtype) -> bool:
-    logger = logging.getLogger(__name__)
```

验证:Qwen3-Embedding-0.6B 在 H200 上以 piecewise CUDA graph + inductor 测试通过。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动限于 `jit_kernel/norm.py` 的 `can_use_fused_inplace_qknorm`。`assume_constant_result` 将该函数的判定固化——dynamo 视其为常量并在编译缓存中内联结果,head_dim/dtype 不同的新编译形状会重新求值,行为与原 eager 求值一致;模块级 logger 对非编译路径无行为变化。若该函数返回值在运行中可能变化(当前基于 head_dim/dtype 的静态判定),`assume_constant_result` 的常量化会掩盖这种变化。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
