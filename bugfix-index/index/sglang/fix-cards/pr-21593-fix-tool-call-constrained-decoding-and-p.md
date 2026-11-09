# PR #21593: Fix tool call constrained decoding and parsing for models with native formats

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21593](https://github.com/sgl-project/sglang/pull/21593) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-04-11 |
| labels | deepseek, run-ci |
| 热度 | 👍 0 · 💬 31 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
配置了模型专属 tool-call-parser 时 `tool_choice="required"` 强套通用 JSON 约束与原生格式冲突、按纯 JSON 解析得 `tool_calls: null`;改用 structural_tag(at_least_one)保留原生格式并修流式解析两处 bug。

## 问题现象
当配置了模型专属 `--tool-call-parser`(如 `kimi_k2`、`deepseekv3`、`qwen25`)且 `tool_choice="required"` 时双重失效:(1) 被强制套上通用 JSON schema 约束,与模型原生 tool call 格式(如 `<|tool_calls_section_begin|>` 特殊 token)冲突;(2) 输出被当纯 JSON 解析,对特殊 token 格式的模型必然失败,返回 `tool_calls: null`。影响所有带原生 tool call 格式的模型:kimi_k2、deepseekv3/v31/v32、qwen25、llama3、mistral、hermes、trinity、interns1、qwen 等。另有流式 bug:无 schema 约束的流式 tool call 产生空参数串(见根因 3)。

## 根因
正文披露三点:
1. required/named 模式下约束走通用 JSON schema,与模型原生格式不兼容;
2. 响应解析按纯 JSON 数组处理,特殊 token 格式解析失败 → `tool_calls: null`;
3. `base_format_detector.py` 流式解析用 `if cur_arguments:` 判断,而空 dict `{}` 在 Python 中为假,导致流式 tool call(无 schema 约束时)产生空参数串;
4. 另 `deepseekv3_detector.py` 的 `structure_info()` 缺少带包裹特殊 token 的完整格式前缀,流式解析器无法识别 tool call。

## 修复方案
按文件:
- `function_call_parser.py`:`get_structure_tag(at_least_one=)`;`get_structure_constraint` 中 required/named 一律用 structural_tag 保留原生格式(strict=true 带工具 schema,strict=false 用空 `{}`,符合 OpenAI 语义);`auto` 行为不变(triggered_tags、at_least_one=False);新增 `$defs` 一致性校验;
- `xgrammar_backend.py`:legacy structural_tag 经 `StructuralTag.from_legacy_structural_tag(...)` 编译,并把 `at_least_one` 传入新 API,使 required 模式可强制产出至少一个 tool call;
- `protocol.py`:`LegacyStructuralTagResponseFormat` 增加 `at_least_one: bool = False`;
- `serving_chat.py`:非流式 `_process_tool_calls` 重排为先尝试模型 parser(required 且 `supports_structural_tag()` 时),否则回退 JSON 数组解析;流式按同样判据在 `JsonArrayParser` 与 `FunctionCallParser` 间选择;
- `base_format_detector.py`:`if cur_arguments:` → `if cur_arguments is not None:`;
- `deepseekv3_detector.py`:`structure_info()` 补全完整前后缀:

```diff
-            begin=">" + name + "\n```json\n",
-            end="\n```<",
-            trigger=">" + name + "\n```json\n",
+            begin="<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
+            + name
+            + "\n```json\n",
+            end="\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
+            trigger="<｜tool▁calls▁begin｜>",
```

测试:300 个 function_call 单测、53 个集成测试(Llama32 + Lfm2 + Qwen25 + Mistral)全过;Kimi-K2.5 实测。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:所有支持 structural_tag 的 detector 在 required/named 模式下的约束与解析路径(输出从通用 JSON 数组变为原生格式;strict=false 时不再做参数级 schema 约束,仅 best-effort);不支持 structural_tag 的 detector 仍走 json_schema 回退;流式空参数行为修复;deepseekv3 流式识别格式变化,依赖旧 `structure_info()` 输出的下游需对齐;`auto` 模式行为明确保持不变。

## 关联
- 正文引用的协议依据:https://platform.openai.com/docs/guides/function-calling(OpenAI function calling 文档,非 issue/PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
