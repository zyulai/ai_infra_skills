# PR #43862: [Bugfix] fix crash in postprocess for null tool args

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43862](https://github.com/vllm-project/vllm/pull/43862) |
| 作者 | william-rom |
| 合并日期 | 2026-06-03 |
| labels | bug, frontend, ready |
| 热度 | 👍 3 · 💬 7 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
tool call 传 arguments:"null" 时 json.loads 得 None 存入消息,渲染 Jinja 模板(如 GLM 5.1)崩溃;后处理中对 None 归一为 {} 修复。

## 问题现象
请求中 tool call 的 `arguments` 为字符串 `"null"` 时,消息进入 chat template 渲染阶段即崩溃:Jinja2 模板对 tool call 调用 `tc.arguments.items()` 会抛出 "'None' has no attribute 'items'"(正文与回归测试描述,举例模型为 GLM 5.1)。触发条件是模型/客户端输出 `argument: "null"` 这类空参数形式。

## 根因
正文披露:`json.loads("null")` 返回 Python `None`,该 None 被直接存进消息;而 `_postprocess_messages` 参数解析处的注释声称 "if arguments is None or empty string, set to {}",代码实际并未覆盖 json 解析结果为 None 的情况——注释与行为不符。

## 修复方案
改 `vllm/entrypoints/chat_utils.py` 的 `_postprocess_messages`,对解析结果为 None 的情况归一为空 dict:

```diff
                 if content := function.get("arguments"):
                     if not isinstance(content, (dict, list)):
-                        function["arguments"] = json.loads(content)
+                        parsed = json.loads(content)
+                        function["arguments"] = parsed if parsed is not None else {}
```

并在 `tests/entrypoints/test_chat_utils.py` 增加回归测试 `test_postprocess_messages_null_arguments_string`,断言 `arguments:"null"` 经后处理后变为 `{}`。作者注明用 Claude Opus 4.7 辅助定位 bug 并起草测试。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 chat_utils 中 tool call `arguments` 为字符串 `"null"` 的后处理路径(原样产出 None,现产出 {}),其他 dict/list/正常 JSON 字符串行为不变(由 diff 推断);模板渲染侧不再收到 None。

## 关联
- #43851

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
