# PR #19818: Fix torch.compile/dynamo crash with Qwen3 QK-norm in piecewise CUDA graph

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jasperjiaguo |
| 合并 | 2026-03-20 |
| 分类 | CUDA Graph / Compiler(classify.py 给出,tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 仅 1 个文件,约 +4/-2 行 |
| 链接 | https://github.com/sgl-project/sglang/pull/19818 |

## 一句话结论

修复 Qwen3(QK-norm)在 `--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler inductor` 下 CUDA graph warmup 时 dynamo `Unsupported` 崩溃,使 PCG+inductor 可用于 QK-norm 模型。

## 动机

Qwen3 系模型使用 QK-norm;开启 piecewise CUDA graph + inductor 编译器后,服务在 CUDA graph warmup 阶段崩溃,报 `torch._dynamo.exc.Unsupported`。根因有二:`can_use_fused_inplace_qknorm()` 在函数体内调用 `logging.getLogger(__name__)`,dynamo trace 该函数时无法处理 `LoggingLoggerVariable` 的字符串比较;函数还经 `_jit_qknorm_module()` 做文件系统 I/O(JIT 编译),dynamo 同样无法 trace。QK-norm 判定函数被编译路径 trace 导致整个 PCG 流程不可用。

## 优化点

两处最小改动,均在 `python/sglang/jit_kernel/norm.py`:

1. 把 `logger = logging.getLogger(__name__)` 从函数体内移到模块级,使 dynamo trace 期间不会遇到 `logging.getLogger()` 调用。
2. 给 `can_use_fused_inplace_qknorm` 加 `@torch.compiler.assume_constant_result` 装饰器,让 dynamo 在 Python 层 eagerly 求值该函数并内联布尔结果,完全避免 trace 进文件系统/JIT 操作。函数本身是纯判定(head_dim 是否支持、架构是否支持 PDL),常量结果假设语义安全;`@cache_once` 保证运行期只算一次。

## 关键改动

```text
python/sglang/jit_kernel/norm.py: logger 移至模块级;can_use_fused_inplace_qknorm 加 assume_constant_result
```

```python
@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm(head_dim: int, dtype: torch.dtype) -> bool:
    # logger 已移至模块级,函数不再含 logging/文件系统调用
    if head_dim not in [64, 128, 256, 512, 1024]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    ...
```

## 性能声明

PR 正文未提供性能对比数据,未注明测试条件之外的量化结果。验证性陈述:用 Qwen3-Embedding-0.6B 在 H200 上配合 piecewise CUDA graph + inductor 测试通过(warmup 不再崩溃)。定性收益是打通 PCG+inductor 在 QK-norm 模型上的可用性,未量化吞吐。

## 验证与回归风险

PR 正文 Accuracy/Benchmarking 章节为空模板,Checklist 未勾选,未见新增单元测试;依赖 CI(run-ci label)。风险极小但存在:`assume_constant_result` 把"当前进程能否用 fused QK-norm"固化为编译期常量,若同一进程内头维度/架构判定结果本可变化(实际由 `cache_once` 已视为不变),行为与既有缓存语义一致;影响面为所有走 JIT fused QK-norm 判定的模型路径(Qwen3 等 QK-norm 模型),非 QK-norm 模型不受影响。

## 关联

同区域:piecewise CUDA graph 编译线——后续 PR #21977(同一作者的 combo_kernels 横向融合)、PR #17706(PCG padding token bugfix)。PR 正文未列出具体编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
