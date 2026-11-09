# PR #37831: [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37831](https://github.com/vllm-project/vllm/pull/37831) |
| 作者 | AAISSJ |
| 合并日期 | 2026-04-01 |
| labels | bug, qwen, ready, tool-calling |
| 热度 | 👍 7 · 💬 11 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
Qwen3Coder 工具解析把所有 anyOf 一律当 object,导致 nullable 参数走错转换分支;新增 _resolve_param_type 按真实类型解析。

## 问题现象
Qwen3CoderToolParser 解析工具调用参数时,对含 `Optional[T]` 参数(Pydantic v2 生成 `anyOf` schema)的工具,参数类型转换出错:例如 `anyOf[string, null]` 的字符串参数会被送进 `json.loads`,解析纯字符串会失败或行为异常(整数 `"5"` 经 `json.loads("5")` 只是碰巧成功);type-as-array(`{"type": ["integer", "null"]}`)与 `$ref` schema(`{"$ref": "#/$defs/ToolInput"}`,无 `"type"` 字段)此前也未被正确处理,`$ref` 会错误落到 `"string"`。

## 根因
正文披露:此前的修复 #36032 对所有 `anyOf` schema 硬编码 `param_type = "object"`。虽然解决了 object 型 anyOf 参数的双重编码问题,但破坏了非 object 类型的 nullable 参数——它们被路由到 `json.loads` 而不是各自的正确转换分支(`int()`、`float()` 等)。根本背景是 Pydantic v2 对每个 `Optional[T]` 字段都会生成 `anyOf` schema(如 `count: Optional[int] = 5` 变为 `{"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 5}`),且 Pydantic 模型输入会产生无 `"type"` 字段的 `$ref` schema。

## 修复方案
在 `vllm/tool_parsers/qwen3coder_tool_parser.py` 新增两个辅助方法,替换原深层嵌套的 if/elif 内联逻辑:
- `_first_non_null_type(type_value)`:静态方法,从标量类型或 type-as-array 中提取第一个非 null 类型,复用于顶层类型解析与 anyOf 变体扫描;
- `_resolve_param_type(param_def)`:按四种情形解析有效类型——直接 `"type"` 字段(含 type-as-array)、`anyOf`/`oneOf` 变体(取第一个非 null 类型)、`$ref` schema(视为 `"object"` 以触发 `json.loads`)、兜底 `"string"`。

`_convert_param_value` 改为调用 `_resolve_param_type`:

```diff
-        if (
-            isinstance(param_config[param_name], dict)
-            and "type" in param_config[param_name]
-        ):
-            param_type = str(param_config[param_name]["type"]).strip().lower()
-        elif (
-            isinstance(param_config[param_name], dict)
-            and "anyOf" in param_config[param_name]
-        ):
-            # anyOf has no top-level "type"; treat as object to trigger json.loads.
-            param_type = "object"
-        else:
-            param_type = "string"
+        if not isinstance(param_config[param_name], dict):
+            return param_value
+
+        param_type = self._resolve_param_type(param_config[param_name])
```

同时新增 `test_extract_tool_calls_anyof_type_conversion` 覆盖 7 种模式(anyof_int/str/array/obj、type_as_array、multi_non_null、ref_param)及对应的 streaming e2e 测试。

## 影响与回归风险
- 影响窗口:正文指明问题自 #36032 的硬编码逻辑起(该 PR 之前 anyOf 直接落 "string"),至本修复前。
- 回归面:`Qwen3CoderToolParser._convert_param_value` 的全部参数类型转换路径(anyOf/oneOf/$ref/type-as-array);`$ref` 一律按 object 处理依赖「被引用类型几乎总是 object」的假设(正文原话),非 object 的 `$ref` 仍会走 `json.loads`;`Qwen3XMLToolParser._get_param_type` 存在同样问题但本 PR 刻意不修,留作 follow-up。

## 关联
- #36032(被本 PR 修正的前置修复)
- #37652(其提出的 `$ref` 处理被本 PR 吸收)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
