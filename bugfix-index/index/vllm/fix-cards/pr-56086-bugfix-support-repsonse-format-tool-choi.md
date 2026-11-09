# PR #56086: [Bugfix] Support repsonse_format + tool_choice=auto

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56086](https://github.com/vllm-project/vllm/pull/56086) |
| 作者 | arpera |
| 合并日期 | 2026-09-27 |
| labels | bug, structured-output, ready, tool-calling |
| 热度 | 👍 0 · 💬 36 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
tool_choice=auto 与 response_format 并用时约束无法共存,改为对两者分别构造结构标签再做 OR 合并。

## 问题现象
请求同时携带 response_format 与 tool_choice=auto 时,两种约束不能按预期同时生效(正文引用 issue #39929 作为报告来源,未直接贴出修复前的错误输出)。正文用 4 个 e2e 复现脚本验证修复后行为:非 strict 工具 + response_format 时仅应用 response_format 并告警;strict 工具 + auto 时输出允许「合法工具调用 OR response_format 约束」;strict 工具无 response_format 时正常产出工具调用。

## 根因
正文披露:此前 vLLM 不支持 response_format + tool_choice=auto 的组合。更早的 #32127 只覆盖 tool_choice=required 情形,且思路是「丢弃 response_format、只允许工具调用」;而 tool_choice=auto 的结构标签默认允许任意文本(plain text),直接套用会使 response_format 分支失去意义,因此两种约束此前无法合并(机制为正文「IMPORTANT」段落说明,即 tool_choice=auto 的 tag 允许纯文本导致约束失效)。

## 修复方案
重写 `vllm/parser/abstract_parser.py` 中 `DelegatingParser._apply_structural_tag`:
- 当 `tool_choice=auto` 且请求带 structured_outputs 时,先用新增的 `structured_outputs_to_format()` 把 StructuredOutputsParams(json/json_object/regex/choice/grammar/structural_tag)转为 xgrammar Format,并用新增的 `_xgrammar_supports()` 校验;
- 构造工具结构标签时,用 `request.model_copy(update={"tool_choice": "required"})` 生成 required 标签,避免 auto 标签允许纯文本;
- 最后把工具标签与输出格式做 OR 合并:
```python
structural_tag = tools_structural_tag
if output_format is not None:
    structural_tag = StructuralTag(
        format=OrFormat(elements=[tools_structural_tag.format, output_format])
    )
```
- xgrammar 不支持的 schema、无结构标签能力的 parser、非 strict 工具等情况回退为仅应用 response_format 并 `warning_once` 告警;`tool_choice=required` + response_format 时告警"structured outputs are ignored because tool_choice forces tool call"。
- harmony.py 删除重复的 `_params_to_final_content` 改用公共 `structured_outputs_to_format`;cohere_command.py 覆盖 `_apply_structural_tag` 直接返回避免二次处理;`adjust_request` 中原本重复的两次 `_tool_parser.adjust_request` 调用合并为一次。
- 新增 tests/parser/test_abstract_parser.py(含回归用例:误用 auto 标签会接受纯文本 "Hello" 必须被拒绝)与 harmony 测试用例。

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:所有经 `DelegatingParser._apply_structural_tag` 的结构化输出+工具调用路径(auto/required、xgrammar 校验失败、lark grammar 转 EBNF、harmony/gpt-oss、cohere);tool_choice=required 且带 response_format 的行为从「静默丢弃」变为「告警后忽略」;lark grammar 现在会先转 EBNF,转换失败抛 VLLMValidationError。

## 关联
- [#39929](https://github.com/vllm-project/vllm/issues/39929)(修复的 issue)
- [#32127](https://github.com/vllm-project/vllm/pull/32127)(前序 required 方案)
- [#39969](https://github.com/vllm-project/vllm/pull/39969)、[#47439](https://github.com/vllm-project/vllm/pull/47439)、[#47394](https://github.com/vllm-project/vllm/pull/47394)(此前被关闭的尝试)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
