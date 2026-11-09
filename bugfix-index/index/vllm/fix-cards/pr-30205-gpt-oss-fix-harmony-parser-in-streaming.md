# PR #30205: [gpt-oss] Fix harmony parser in streaming responses

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30205](https://github.com/vllm-project/vllm/pull/30205) |
| 作者 | AlonKejzman |
| 合并日期 | 2025-12-22 |
| labels | frontend, ready, gpt-oss |
| 热度 | 👍 5 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
speculative decoding 下流式响应最后一批 token 丢失;channel 为空但有 delta 文本时按 final 处理。

## 问题现象
在 speculative decoding 场景下,流式响应会丢失最后的 final token(对应 issue #30204,正文目的即 "Fix missed final streaming tokens in speculative decoding")。正文未展开更详细的症状与复现步骤;测试计划为修复后重跑复现脚本,测试结果为 "Final tokens are indeed returned as expected."。

## 根因
正文未披露根因(正文仅一句话说明目的)。(由 diff 推断)diff 新增注释描述了机制:当一次生成多个 token 且其中包含最后一个 token 时,delta 文本非空,但 harmony 解析器的 current_channel 已回到空(start 状态),导致这段 delta 没有可归属的 channel;修复即在 channel 为空但 delta_text 非空时按 `"final"` 处理。

## 修复方案
改动仅一处,位于 `vllm/entrypoints/openai/serving_chat.py` 的 `chat_completion_stream_generator`:在读取 `harmony_parser.current_channel` 与 `current_recipient` 之后,若 `cur_channel` 为空但 `delta_text` 非空,则将 `cur_channel` 置为 `"final"`,使这批 token 作为 final 内容继续走原有输出逻辑。

```python
+                        # handle the case where several tokens where generated at once
+                        # including the final token, leading to a delta in the text
+                        # but the current channel to be empty (start state)
+                        if not cur_channel and delta_text:
+                            cur_channel = "final"
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:按 diff,仅影响 serving_chat 流式生成器中 harmony 解析分支的 channel 判定这一处:此前「channel 为空且有文本」的 delta 会被丢弃或无处归属,现在统一按 final 通道输出;下游依赖 `cur_channel` 的输出分支行为随之改变。

## 关联
- https://github.com/vllm-project/vllm/issues/30204

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
