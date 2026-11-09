# PR #25681: Revert "[Bug] Dynamo Unsupported due to `BasevLLMParameter.torch_function` calling disabled super()"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25681](https://github.com/vllm-project/vllm/pull/25681) |
| 作者 | mgoin |
| 合并日期 | 2025-09-25 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
整体 revert #25613:移除 torch<2.8 时 BasevLLMParameter.__torch_function__ 返回 NotImplemented 的分支,恢复直接调 super()。

## 问题现象
正文未描述具体现象,仅说明本 PR 是对 vllm-project/vllm#25613 的整体 revert。从被 revert 的标题可知,#25613 当时针对的问题是「Dynamo Unsupported due to `BasevLLMParameter.torch_function` calling disabled super()」,本次 revert 撤销了那个修复。

## 根因
正文未披露根因(正文只有一行 "Reverts vllm-project/vllm#25613")。由 diff 推断:#25613 曾在 `BasevLLMParameter.__torch_function__` 中加入「torch 版本低于 2.8.0 时记录 warning 并返回 `NotImplemented`」的分支,以规避 TorchDynamo 对调用 `super().__torch_function__` 的问题;本 revert 认为该分支不再需要,恢复无条件走 `super().__torch_function__`。

## 修复方案
修改 `vllm/model_executor/parameter.py` 的 `BasevLLMParameter.__torch_function__`:删除 `is_torch_equal_or_newer("2.8.0")` 检查及其 `logger.warning_once` + `return NotImplemented` 早退分支;同时移除不再使用的 `from vllm.utils import is_torch_equal_or_newer` 导入。修复后该 classmethod 只保留 kwargs 兜底并直接调用:

```diff
-        if not is_torch_equal_or_newer("2.8.0"):
-            logger.warning_once(
-                "Torch %s detected (<2.8.0): returning NotImplemented in "
-                "BasevLLMParameter.__torch_function__ to avoid potential "
-                "TorchDynamo issues.",
-                torch.__version__,
-            )
-            return NotImplemented
-
         if kwargs is None:
             kwargs = {}
         return super().__torch_function__(func, types, args, kwargs)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`vllm/model_executor/parameter.py` 中 `BasevLLMParameter.__torch_function__` 的所有 torch<2.8.0 路径——即 #25613 想保护、而本 revert 重新暴露给 TorchDynamo 的场景;若 Dynamo 旧版对 `super().__torch_function__` 的问题在旧 torch 上仍然存在,compile 路径可能再次触发(由 diff 推断);torch>=2.8 路径行为不变。

## 关联
- vllm-project/vllm#25613(被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
