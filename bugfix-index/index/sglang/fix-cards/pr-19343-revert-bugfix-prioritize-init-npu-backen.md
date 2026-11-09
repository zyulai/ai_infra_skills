# PR #19343: Revert "bugfix: prioritize init_npu_backend to fix various initialization bugs"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19343](https://github.com/sgl-project/sglang/pull/19343) |
| 作者 | iforgetmyname |
| 合并日期 | 2026-02-25 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #17652:撤销把 init_npu_backend 提前到 server_args 的改法,恢复 model_runner 模块级初始化 NPU 后端。

## 问题现象
正文仅一句 "Reverts sgl-project/sglang#17652",未描述具体现象。被回退的原 PR 标题为 "bugfix: prioritize init_npu_backend to fix various initialization bugs",本 PR 撤销了那次"提前初始化"的改动,说明该改法引入了需要回退的问题;具体症状、复现条件与影响面正文均未给出。

## 根因
正文未披露根因,仅说明本 PR 是对 #17652 的 revert。(由 diff 推断)机制:回退把 `init_npu_backend()` 的调用从 `server_args.py` 的 `_handle_npu_backends()` 中移除(该函数内只保留 `set_default_server_args` 导入与调用),同时在 `model_runner.py` 模块顶层恢复 `_is_npu` 分支:

```diff
+if _is_npu:
+    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
+
+    init_npu_backend()
```

即 NPU 后端初始化回到 model_runner 导入时执行,而非在 server_args 处理阶段提前执行;暗示 #17652 的提前初始化路径引入了回归。

## 修复方案
整体 revert #17652,改动两个文件:
- `python/sglang/srt/model_executor/model_runner.py`:在 `_is_cpu_amx_available`/`_is_cpu_arm64` 定义之后、`MLA_ATTENTION_BACKENDS` 之前,恢复模块级 `if _is_npu: init_npu_backend()`。
- `python/sglang/srt/server_args.py`:`_handle_npu_backends()` 中删除 `init_npu_backend` 的导入与调用,仅保留 `set_default_server_args`。

关键 diff(server_args.py):
```diff
         if self.device == "npu":
-            from sglang.srt.hardware_backend.npu.utils import (
-                init_npu_backend,
-                set_default_server_args,
-            )
-
-            init_npu_backend()
+            from sglang.srt.hardware_backend.npu.utils import set_default_server_args
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #17652 合入后至本回退落地之间。
- 回归面:仅影响 NPU(device == "npu")部署路径——后端初始化时机从 server_args 处理阶段移回 model_runner 模块导入时。依赖"初始化先于 server_args"这一顺序的 NPU 代码路径会回到旧时序;非 NPU 平台不受影响(分支有 `_is_npu` 保护)。

## 关联
- [sgl-project/sglang#17652](https://github.com/sgl-project/sglang/pull/17652)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
