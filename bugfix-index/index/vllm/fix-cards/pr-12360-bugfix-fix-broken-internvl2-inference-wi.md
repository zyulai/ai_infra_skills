# PR #12360: [Bugfix] Fix broken internvl2 inference with v1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12360](https://github.com/vllm-project/vllm/pull/12360) |
| 作者 | Isotr0py |
| 合并日期 | 2025-01-23 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
v1 引擎下 InternVL2 因 modality_group_func 对空 modalities 取首元素越界而崩溃,补上空分支返回空串修复。

## 问题现象
v1 引擎跑 InternVL2 时 EngineCore 直接抛异常崩溃:调用链 `execute_model → _execute_encoder → group_mm_inputs_by_modality → modality_group_func`,在 `vllm/multimodal/utils.py:508` 的 `return list(mm_input.modalities)[0]` 处抛 `IndexError: list index out of range`,随后 worker 收到 fatal signal,进程整体关停(日志时间 2025-01-23 21:30:09)。

## 根因
正文仅说明「InternVL2 inference on v1 is broken due to `modality_group_func`」,未展开机制。由 diff 推断:来自 legacy pipeline 的 mm_input 其 modalities 为空(diff 内 FIXME 注释写明 "Modality of mm_input from legacy pipeline is empty"),原代码只处理了 len>1 与"取第一个"两种情况,未处理 len==0,直接 `list(...)[0]` 越界。

## 修复方案
`vllm/multimodal/utils.py` 的 `modality_group_func` 补上空分支:

```diff
-        # Otherwise return the modality string
-        return list(mm_input.modalities)[0]
+        elif len(mm_input.modalities) == 1:
+            return list(mm_input.modalities)[0]
+
+        # FIXME(Isotr0py): Modality of mm_input from legacy pipeline is empty,
+        # this is used to make InternVL with legacy pipeline still work with v1.
+        else:
+            return ""
```

空 modality 时返回空字符串分组键,使 InternVL legacy pipeline 在 v1 下仍能工作。

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:`group_mm_inputs_by_modality` 的分组键逻辑,影响所有走 v1 encoder 路径的多模态模型;modalities 非空的既有行为(len>1 用 `id(mm_input)`,len==1 用 modality 字符串)保持不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
