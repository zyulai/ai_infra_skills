# PR #4931: Fix sglang frontend's incorrect dependency on torch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4931](https://github.com/sgl-project/sglang/pull/4931) |
| 作者 | seplos |
| 合并日期 | 2025-03-30 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

前端 utils 顶层误 import 依赖 torch 的 srt 模块;把 kill_process_tree 的导入移回函数内。

## 问题现象

正文未描述具体现象,仅写「Resolve #4929」。从标题与修改内容看:前端 `sglang/utils.py` 对 torch 产生了不正确的依赖,导致前端场景 import 该模块时会连带加载依赖 torch 的 `sglang.srt.utils`(由 diff 推断)。

## 根因

正文标题披露问题是「sglang frontend's incorrect dependency on torch」,Modifications 说明修复方式是把 `kill_process_tree` 的 import 移回函数内。即此前 `from sglang.srt.utils import kill_process_tree` 被提升到了 `sglang/utils.py` 模块顶层,而 `sglang.srt.utils` 依赖 torch,顶层 import 使整个前端 utils 被拖入对 torch 的硬依赖;具体由哪次改动引入,正文未说明。

## 修复方案

改 `python/sglang/utils.py` 两处:

```diff
-from sglang.srt.utils import kill_process_tree
```

删除模块顶层的上述 import,并将其移回 `terminate_process` 函数内部,延迟到真正终止进程时才导入:

```diff
     Terminate the process and automatically release the port.
     """
+    from sglang.srt.utils import kill_process_tree
+
     kill_process_tree(process.pid)
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 `sglang/utils.py` 的导入结构变化;`terminate_process` 的功能行为不变,只是导入时机变为函数内延迟导入;其他使用 `kill_process_tree` 的路径不受影响。前端在不加载 torch 的环境下 import `sglang.utils` 不再失败。

## 关联

- https://github.com/sgl-project/sglang/issues/4929

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
