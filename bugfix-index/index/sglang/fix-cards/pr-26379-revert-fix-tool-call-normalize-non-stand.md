# PR #26379: Revert "fix(tool_call): normalize non-standard JSON Schema types in tool params"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26379](https://github.com/sgl-project/sglang/pull/26379) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-05-26 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
整体回滚 #23476:移除 tool 参数 JSON Schema 非标准 type 归一化层与 RecursionError 处理,恢复原校验行为。

## 问题现象
正文未描述具体现象,仅一句 "Reverts sgl-project/sglang#23476",连回滚原因也未说明。pack 中的 CI 状态显示本 PR 自身的 Base 与 Extra 两个测试 run 均为失败(:x:)。

## 根因
正文未披露根因。

(由 diff 推断)本 PR 撤销 #23476 引入的兼容层:`_validate_request` 在 `Draft202012Validator.check_schema` 之前调用的 `normalize_json_schema_types()`(把 DB/ORM 风格的 `"type"` 别名如 varchar/enum/int 改写为标准 JSON Schema 类型,并以 `except RecursionError` 把手写循环 schema 挡成 400 而非 500)被整体移除;同时删除 `function_call/utils.py` 中约 150 行的别名表(`_JSON_SCHEMA_TYPE_ALIASES`、`_PREFIX_RULES` 等)与递归遍历实现,以及对应单测文件。回滚的具体原因正文未给出。

## 修复方案
- `python/sglang/srt/entrypoints/openai/serving_chat.py`:`_validate_request` 恢复为仅 `Draft202012Validator.check_schema(tool.function.parameters)` 校验,删除 normalize 调用与 `except RecursionError` 分支;import 收敛回 `get_json_schema_constraint`:

```diff
-                normalize_json_schema_types(tool.function.parameters)
                 Draft202012Validator.check_schema(tool.function.parameters)
             except SchemaError as e:
                 return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"
-            except RecursionError:
-                return (
-                    f"Tool {i} function 'parameters' schema is too deeply nested "
-                    "or contains a cycle."
-                )
```

- `python/sglang/srt/function_call/utils.py`:删除 `_STANDARD_JSON_SCHEMA_TYPES`、`_JSON_SCHEMA_TYPE_ALIASES`、`_PREFIX_BOUNDARY_CHARS`、`_PREFIX_RULES`、`_matches_type_prefix`、`_normalize_single_type`、`normalize_json_schema_types`。
- 删除 `test/registered/unit/function_call/test_normalize_json_schema_types.py`(约 390 行单测)。

## 影响与回归风险
- 影响窗口:自 #23476 合并起至本回滚。
- 回归面:OpenAI serving chat 的 tool `parameters` schema 校验路径——DB/ORM 风格的非标准 type 不再被改写为标准类型,会被 `check_schema` 拒绝(返回 400);`RecursionError` 转 400 的防护一并移除,手工构造的循环 schema 恢复为未捕获异常路径(被删代码注释称其原为避免 500)(由 diff 推断)。#23476 面向的 DB/ORM 工具 schema 兼容场景随之失效。

## 关联
- 回滚对象:sgl-project/sglang#23476

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
