# PR #38751: Revert "[Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params (#37831)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38751](https://github.com/vllm-project/vllm/pull/38751) |
| 作者 | khluu |
| 合并日期 | 2026-04-01 |
| labels | bug, qwen, ready, tool-calling |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
#37831 的 anyOf/oneOf 类型解析修复在 main 上弄坏测试,本 PR 整体 revert 该 commit,恢复旧内联逻辑。

## 问题现象
正文未描述具体现象,仅说明被 revert 的 commit(582340f273666d768630aca65314dd104be41b14,即 #37831)在 main 上弄坏了一个测试(breaks a test on main),具体是哪个测试、以何种方式失败,正文未给出。

## 根因
正文未披露根因——只陈述 revert 的事实的动机(该 commit breaks a test on main),没有解释 #37831 的改动为何、在哪条路径上破坏了测试。由 diff 推断:revert 恢复了 #37831 之前的内联类型解析逻辑(直接 `"type"` 字段→取字符串;`anyOf`→一律 `"object"`;否则 `"string"`),并删除 `_first_non_null_type`/`_resolve_param_type` 两个辅助方法及新增的两条 anyOf/streaming 测试。

## 修复方案
整体 revert #37831,涉及两个文件:
1. `vllm/tool_parsers/qwen3coder_tool_parser.py`:删除 `_first_non_null_type` 与 `_resolve_param_type`;`_convert_param_value` 恢复为原先的 if/elif/else 内联逻辑(含 `anyOf` 硬编码 `param_type = "object"` 的注释行);
2. `tests/tool_parsers/test_qwen3coder_tool_parser.py`:删除 `test_extract_tool_calls_anyof_type_conversion` 与 `test_extract_tool_calls_anyof_type_conversion_streaming` 两个测试。

```diff
-    def _resolve_param_type(self, param_def: dict) -> str:
-        """Resolve the effective type string from a parameter definition.
-        ...
-        """
-        ...
-        return "string"
```

测试计划:CI 应在该 revert 生效后通过(CI should pass with the revert applied)。

## 影响与回归风险
- 影响窗口:#37831 合并日起,至本 revert 合并日(两者同为 2026-04-01)。
- 回归面:revert 意味着 #37831 所修的 nullable 参数(anyOf/oneOf/type-as-array/$ref)类型解析问题在 Qwen3CoderToolParser 上重新存在——非 object 类型的 `Optional[T]` 参数会再次被路由到 `json.loads`(由 diff 推断);若 CI 未暴露本次失败的测试,后续再修时需同时处理那个被破坏的测试。

## 关联
- #37831(被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
