# PR #30083: Revert "Fix wrong RMSNorm fallback to old Flashinfer CUDA kernel when in PCG"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30083](https://github.com/sgl-project/sglang/pull/30083) |
| 作者 | ch-wan |
| 合并日期 | 2026-07-04 |
| labels | sgl-kernel |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
整体回滚 PR #29702(RMSNorm 在 PCG 下错误回退到旧 Flashinfer CUDA kernel 的「修复」),恢复 sgl-kernel norm 分发与 layernorm.py 的原实现。

## 问题现象
正文未描述具体现象,仅一句「Reverts sgl-project/sglang#29702」。即被回滚的 PR #29702(标题为 Fix wrong RMSNorm fallback to old Flashinfer CUDA kernel when in PCG)本身引发问题,需要整体撤销。CI 状态显示本 PR 的 Base/Extra 测试当时亦有失败记录(Run #28691145877 / #28691145854)。

## 根因
正文未披露根因。(由 diff 推断) 本 PR 把两处改动退回:
- `sgl-kernel/python/sgl_kernel/elementwise.py`:重新引入 `not torch.compiler.is_dynamo_compiling()` 条件,并恢复一段注释,说明 FlashInfer norm 路径在 `torch.compile(..., fullgraph=True)` 下不安全——Dynamo 会追踪进 FlashInfer 的 JIT 模块加载路径,调用 `Path.exists()`/`os.stat()` 等不可追踪操作导致编译失败,故 tracing 时回退内部实现,属临时 workaround,并引用 flashinfer-ai/flashinfer#2734 与 #2733。也就是说 #29702 移除该判断后,PCG/torch.compile 场景的 RMSNorm 分发方式被证明有问题。
- `python/sglang/srt/layers/layernorm.py`:`#29702` 曾把 `from sgl_kernel import ...` 的四个 norm 函数经 `register_custom_op_from_extern` 包装为 custom op;本 PR 恢复为直接 import。

## 修复方案
按 revert 性质原样回退 #29702 的改动,涉及两个文件:

- `sgl-kernel/python/sgl_kernel/elementwise.py`:`rmsnorm`、`fused_add_rmsnorm`、`gemma_rmsnorm`、`gemma_fused_add_rmsnorm` 四个函数的 FlashInfer 分发条件重新加上 `and not torch.compiler.is_dynamo_compiling()`,命中时仍走 `_flashinfer_norm.*`,否则走 `_*_internal` 内部实现。
- `python/sglang/srt/layers/layernorm.py`:删除 `register_custom_op_from_extern` 包装,恢复:

```diff
-    from sgl_kernel import fused_add_rmsnorm as _sgl_fused_add_rmsnorm
-    ...
-    rmsnorm = register_custom_op_from_extern(
-        _sgl_rmsnorm, op_name="sgl_rmsnorm", out_shape="input"
-    )
+    from sgl_kernel import (
+        fused_add_rmsnorm,
+        gemma_fused_add_rmsnorm,
+        gemma_rmsnorm,
+        rmsnorm,
+    )
```

## 影响与回归风险
- 影响窗口:PR #29702 合并起至本 PR 合并(2026-07-04)之间使用 main 分支的版本。
- 回归面:`sgl-kernel/python/sgl_kernel/elementwise.py` 四个 norm 算子的分发逻辑与 `python/sglang/srt/layers/layernorm.py` 的 import/注册方式。torch.compile(fullgraph)/PCG 场景恢复为 tracing 时走内部实现;同时 #29702 想解决的「PCG 下错误回退到旧 Flashinfer CUDA kernel」问题随之恢复存在(即回到回滚前的行为),后续需等 FlashInfer 上游修复(issue #2734 / PR #2733)再重新处理。custom op 注册方式恢复原样,依赖 `sgl_rmsnorm` 等 op 名称的下游需注意。

## 关联
- PR #29702(被回滚的改动): https://github.com/sgl-project/sglang/pull/29702
- flashinfer-ai/flashinfer#2734、flashinfer-ai/flashinfer#2733(见 diff 注释引用,非正文列出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
