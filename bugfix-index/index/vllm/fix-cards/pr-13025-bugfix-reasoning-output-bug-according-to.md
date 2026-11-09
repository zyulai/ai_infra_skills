# PR #13025: [Bugfix]: Reasoning output bug according to the chat template change

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#13025](https://github.com/vllm-project/vllm/pull/13025) |
| 作者 | gaocegege |
| 合并日期 | 2025-02-11 |
| labels | frontend, ready |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论

DeepSeek-R1 新 chat 模板不再输出 `<think>`,旧 reasoning parser 解析错位;改为兼容有无 `<think>` 两种格式。

## 问题现象

DeepSeek 于前一天更新了 DeepSeek-R1 的 chat template(HuggingFace 提交 8a58a132790c9935686eb97f042afa8013451c9f):模板在 prompt 中加入 `<think>` token,标记首段为思考过程,因此**最终生成的输出不再包含 `<think>`**,只会在中途输出 `</think>`。而 vLLM 的 DeepSeek-R1 reasoning parser 此前假定模型输出一定以 `<think>` 开头并以 `</think>` 结束,导致在这类输出上无法正确拆分 `reasoning_content` 与 `content`(reasoning 输出错误,fixes issue #12999)。

## 根因

PR 正文披露:根因是 DeepSeek 官方 chat template 变更——`<think>` 被移入 chat template,模型输出不再自带 `<think>` 前缀;旧 parser 的流式与非流式分支都要求 `<think>`/`</think>` 成对出现才提取 reasoning content,因而对「只有 `</think>`」的新格式输出解析失败。

## 修复方案

改 `vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py`,同时兼容两种风格(`<think>...</think>...` 与 `...</think>...`):

- 流式:在「previous 和 delta 中都没有 `<think>`」的分支里,改为继续检查 `</think>` 是否出现在 delta 或 previous 中,分别做切分/切换到 content;没有 `</think>` 时视为 reasoning 内容继续。

```diff
-            # No <think> in previous or delta, reasoning content continues.
-            return DeltaMessage(content=delta_text)
+            # No <think> in previous or delta, also need to check for </think>.
+            if self.think_end_token_id in delta_token_ids:
+                end_index = delta_text.find(self.think_end_token)
+                reasoning_content = delta_text[:end_index]
+                content = delta_text[end_index + len(self.think_end_token):]
+                return DeltaMessage(reasoning_content=reasoning_content,
+                                    content=content if content else None)
+            elif self.think_end_token_id in previous_token_ids:
+                return DeltaMessage(content=delta_text)
+            else:
+                return DeltaMessage(reasoning_content=delta_text)
```

- 非流式:只要输出含 `</think>` 就假定 reasoning 在开头;缺失 `<think>` 时先补上再走原 regex 提取。
- 同时更新 `tests/entrypoints/openai/reasoning_parsers/test_deepseekr1_reasoning_parser.py`(新增 with_think/无 think 用例)与 `examples/online_serving/openai_chat_completion_with_reasoning.py` 的打印文案。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 DeepSeek-R1 reasoning parser 的流式与非流式解析路径及其测试、示例脚本;使用其他 reasoning parser 的模型不受影响。若某模型输出中 `</think>` 出现在非开头位置,新「假定 reasoning 在开头」的假设可能仍不适用。

## 关联

- https://github.com/vllm-project/vllm/issues/12999
- https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
