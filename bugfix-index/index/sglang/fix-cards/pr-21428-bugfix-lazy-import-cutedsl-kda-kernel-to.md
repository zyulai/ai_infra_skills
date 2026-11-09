# PR #21428: [Bugfix] Lazy-import CuteDSL KDA kernel to fix AMD/ROCm startup crash

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21428](https://github.com/sgl-project/sglang/pull/21428) |
| 作者 | hubertlu-tw |
| 合并日期 | 2026-03-25 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论

#21203 顶层 import CuteDSL KDA 内核,链式拉起 CUDA-only 包致 AMD 启动即崩;改为 `is_cuda()` 守卫内懒加载修复。

## 问题现象

在 AMD GPU 或其他非 CUDA 加速器上,启动任何带 linear attention 的模型(如 Qwen3.5)都会立刻崩溃,报错:

```
ModuleNotFoundError: No module named 'cuda'
```

服务器无法启动,影响所有 AMD/ROCm 上的 linear attention 模型。

## 根因

按 PR 正文:#21203(`[KDA] Support CuTeDSL KDA decode kernel`)在 `kda_backend.py` 顶层 import 了 `CuteDSLKDAKernel`。该导入链为 `kda_backend.py → kda_cutedsl.py → cutedsl_kda.py → import cuda.bindings.driver`,而 `cuda.bindings.driver` 是 CUDA-only 的包,AMD/ROCm 环境不存在,顶层 import 无条件执行,导致启动即抛 `ModuleNotFoundError`。

## 修复方案

改 `python/sglang/srt/layers/attention/linear/kda_backend.py`:把顶层 `from ...kda_cutedsl import CuteDSLKDAKernel` 移到 `__init__` 内 `is_cuda()` 守卫分支中懒加载——只有显式选择 CuTeDSL decode backend 且平台为 CUDA 时才 import:

```diff
-from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
-    CuteDSLKDAKernel,
-)
 ...
         elif decode_backend.is_cutedsl():
             if not is_cuda():
                 raise ValueError("KDA CuTe DSL backend requires CUDA")
+            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
+                CuteDSLKDAKernel,
+            )
+
             self.decode_kernel = CuteDSLKDAKernel()
```

非 CUDA 平台永远走不到该代码路径,CUDA-only 模块不再被导入。

## 影响与回归风险

- 影响窗口:#21203 合入之后、本修复之前,AMD/ROCm 上启动 linear attention 模型即崩。
- 回归面:KDA backend 的 import 时序。CUDA 平台行为不变——正文 Test plan 中 "CUDA 平台不受影响" 一项为未勾选状态,即作者尚未在 CUDA 上验证,选择 CuTeDSL backend 时 import 仍会发生(该项为待验证事项);非 CUDA 平台上若显式选择 CuTeDSL backend,仍会按 `raise ValueError("KDA CuTe DSL backend requires CUDA")` 报错。作者已验证 Qwen3.5-397B-A17B-MXFP4 在 AMD MI355X(4×GPU,`--attention-backend aiter`)上修复后可正常启动。

## 关联

- #21203(引入顶层 import 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
