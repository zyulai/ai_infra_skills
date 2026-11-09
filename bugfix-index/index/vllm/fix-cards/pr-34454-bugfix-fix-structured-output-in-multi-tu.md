# PR #34454: [Bugfix]: Fix structured output in multi-turn gpt-oss

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34454](https://github.com/vllm-project/vllm/pull/34454) |
| 作者 | bbrowning |
| 合并日期 | 2026-02-13 |
| labels | bug, gpt-oss, ready, structured-output, v1 |
| 热度 | 👍 5 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
多轮 gpt-oss 误匹配先前消息 final 标记,过早施 grammar 掩码致输出为空;回溯遇 `<|end|>` 即停。

## 问题现象
多轮对话中 gpt-oss 使用结构化输出(如 `json_object` response_format)会产生空或无效输出。复现(正文 Test Plan,即 #32791 的 case 3 手工复现):`vllm serve openai/gpt-oss-20b --tool-call-parser openai --enable-auto-tool-choice`,随后发送含先前 assistant 消息的多轮请求并指定 `response_format: { "type": "json_object" }`、`max_tokens: 128`、`temperature: 0`。修复前响应为 `"content": null`、`"finish_reason": "stop"`;修复后返回 `"content": "{\"response\":\"bye\"}"` 并带 reasoning。

## 根因
正文披露:`gptoss_reasoning_parser` 中检测「模型何时结束 reasoning、开始输出 final 通道内容」的逻辑,会误匹配多轮场景中先前消息里的 final 通道消息;实践中导致 vLLM 在这类多轮对话里对模型全部输出过早应用 grammar bitmask,模型偏离其训练的 Harmony 格式,产生空或无效输出。正文还说明:该 bug 在 vLLM v0.13.0 之前就存在,但当时 gpt-oss 多轮对话状态处理缺失关键 token,恰好使先前对话不匹配这些 token id 检查而未触发;多轮对话状态修复后,structured output 才开始踩中此 bug。

## 修复方案
`vllm/reasoning/gptoss_reasoning_parser.py`:
- `__init__` 新增 `self.eom_token_id = self.model_tokenizer.vocab["<|end|>"]`;
- `is_reasoning_end` 从后往前搜索 end 序列 prefix 时,一旦遇到先前消息的 `<|end|>` token 即提前返回 `False`,不再向更早的消息回溯;只有当前轮实际出现 final 通道标记才判定 reasoning 结束;
- 新增多轮单元测试 `MULTI_TURN_CONTENT`(期望 `is_reasoning_end: False`),mock tokenizer fixture 补充 `vocab = {"<|end|>": 6}` 属性。

```python
         for i in range(len(input_ids) - len(end_token_ids_prefix), -1, -1):
+            if input_ids[i] == self.eom_token_id:
+                # We looped backwards far enough to find the end of a previous message,
+                # which means we have searched the entirety of the current message
+                # and can exit early without searching further back into prior
+                # messages of the conversation.
+                return False
```

测试按 TDD 进行:先让新测试失败,修复后 `tests/reasoning/test_gptoss_reasoning_parser.py`、`tests/v1/structured_output/test_gptoss_structural_tags.py`、`tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` 全部通过;手工 curl 复现也恢复预期输出。

## 影响与回归风险
- 影响窗口:bug 在 vLLM v0.13.0 之前即存在,多轮对话状态修复后才实际暴露(正文表述)。
- 回归面:按 diff,`is_reasoning_end` 的反向搜索深度被截断于最近一条 `<|end|>`;影响依赖该判定的 gpt-oss structured output(grammar bitmask 应用时机)与 reasoning 解析路径;mock tokenizer 测试新增 `vocab` 属性依赖。

## 关联
- Fixes #32791(手工复现 case 3 出自该 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
