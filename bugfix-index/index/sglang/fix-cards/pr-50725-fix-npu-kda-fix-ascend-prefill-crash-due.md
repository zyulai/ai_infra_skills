# PR #50725: [Fix] [NPU] kda: Fix Ascend prefill crash due to missing kernel flag

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#50725](https://github.com/sgl-project/sglang/pull/50725) |
| 作者 | hanwlax |
| 合并日期 | 2026-11-17 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论

#50102 在共享 KDA 元数据初始化中新增了对 `wants_int64_query_start_loc` 属性的读取,而 Ascend KDA prefill kernel 缺少该属性,导致初始化直接抛 `AttributeError`;本 PR 在 `_AscendKDAExtendKernel` 上补上 `wants_int64_query_start_loc = False` 修复崩溃。

## 问题现象

Ascend(NPU)上运行 KDA prefill 时,共享 KDA 元数据初始化阶段失败并抛出 `AttributeError`。触发来源是 #50102 引入的行为:该 PR 在共享元数据初始化路径中读取 kernel 的 `wants_int64_query_start_loc` 能力标记,用于决定是否把 `query_start_loc` 转成 int64;Ascend KDA prefill kernel 类上没有定义这个属性,Python 属性查找失败即崩溃。正文未附完整堆栈。

## 根因

正文披露:根因是 #50102 为共享 KDA 元数据初始化添加了 `wants_int64_query_start_loc` 的读取,而 `_AscendKDAExtendKernel` 没有声明该能力属性,初始化因此以 `AttributeError` 失败。属于跨 PR 的能力标记接口不齐:其他 KDA kernel 实现声明了该 flag,Ascend 实现被遗漏。

## 修复方案

在 `python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py` 的 `_AscendKDAExtendKernel` 类上补写 `wants_int64_query_start_loc: bool = False`。

取值依据(正文):Ascend kernel 在调用 `chunk_kda_fwd` 之前已经自行把 `query_start_loc` 转换为 int64,因此可以跳过共享元数据初始化中的这一转换,`False` 是语义正确的取值而非仅为消除报错。

关键 diff 摘录:

```diff
@@ -21,6 +21,8 @@ class _AscendKDAExtendKernel:

     uses_state_checkpoints: bool = False

+    wants_int64_query_start_loc: bool = False
+
     def extend(
         self,
         q: torch.Tensor,
```

## 影响与回归风险

- 影响窗口:仅 Ascend NPU 上的 KDA prefill 路径;修复后初始化不再抛 `AttributeError`,kernel 计算本身不变(正文明确"the change supplies a missing capability flag and keeps the existing Ascend kernel computation")。
- 回归风险:极小——只新增一个类属性声明,不改任何计算路径;`False` 取值与 kernel 内已有的 int64 转换行为一致,`query_start_loc` 的 dtype 语义不变。
- 正文注明:该 commit 无 accuracy 测试结果(无 kernel 计算变化)、无 benchmark/profiling 结果;Checklist 中"Add unit tests"未勾选,无新增单测。CI 状态:Base 与 AMD ROCm 10 通过,Extra 失败(与本次改动相关的失败原因 pack 未提供)。

## 关联

- 修复的引入方:#50102(新增 `wants_int64_query_start_loc` 读取)。
- 涉及文件:`python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py`(`_AscendKDAExtendKernel`)、下游 `chunk_kda_fwd` 调用。
- CI run:[Base 通过 #43824628157](https://github.com/sgl-project/sglang/actions/runs/43824628157)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
