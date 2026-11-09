# PR #17369: [Bugfix] add qwen3 reasoning-parser fix content is None when disable …

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17369](https://github.com/vllm-project/vllm/pull/17369) |
| 作者 | mofanke |
| 合并日期 | 2025-04-29 |
| labels | documentation, ready |
| 热度 | 👍 6 · 💬 9 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
新增 qwen3 reasoning-parser:enable_thinking=False 时 content 不再为 None,并正确拆分推理内容与正文。

## 问题现象
以 `--enable-reasoning` 跑 Qwen3 时缺少 qwen3 专用解析器,标题指出的现象是禁用推理(disable)时 content 为 None。正文给出复现命令 `python3 -m vllm.entrypoints.openai.api_server --model Qwen3-32B -tp 4 --enable-reasoning --reasoning-parser qwen3`,以及 `chat_template_kwargs: {"enable_thinking": False}` 的 chat.completions 测试请求。FIX #17357。

## 根因
正文未披露根因机制,仅说明 FIX #17357、新增 qwen3 parser,逻辑改自 gaocegege/vllm 项目的 deepseek_r1_reasoning_parser.py(原作者 @gaocegege)。由 diff 推断:新 parser 在模型输出不含 `<think>`/`</think>` 时返回 `(None, model_output)`,即整段输出保留为 content,从而避免禁用思考输出时 content 被置 None。

## 修复方案
新增 `vllm/reasoning/qwen3_reasoning_parser.py`,`@ReasoningParserManager.register_module("qwen3")`:按 `<think>`/`</think>` 标记(流式走 token id 快速路径)拆分 reasoning_content 与 content;`vllm/reasoning/__init__.py` 导出 Qwen3ReasoningParser;新增 `tests/reasoning/test_qwen3_reasoning_parser.py`(10 个用例,含 without_think、only_open_tag、multiline、streaming 等);`docs/source/features/reasoning_outputs.md` 模型表加入 Qwen3 行(支持 guided_json/guided_regex)。

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:以新增文件为主;`vllm/reasoning/__init__.py` 导出列表与文档表各加一项,不影响既有 deepseek_r1、granite 解析器路径。

## 关联
[#17357](https://github.com/vllm-project/vllm/issues/17357)、[gaocegege/vllm deepseek_r1_reasoning_parser.py](https://github.com/gaocegege/vllm/blob/b6339acf07b3fcf7de71c04669f6b042f3bd1823/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
