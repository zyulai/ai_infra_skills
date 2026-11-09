# PR #29972: [Bugfix] Fix parse_output_message crash on commentary with no recipient

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29972](https://github.com/vllm-project/vllm/pull/29972) |
| 作者 | strinczer |
| 合并日期 | 2025-12-05 |
| labels | frontend, ready, gpt-oss |
| 热度 | 👍 3 · 💬 3 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
gpt-oss 生成无 recipient 的 commentary 前导文本时解析崩溃、流提前中断,改为按 reasoning 内容解析修复。

## 问题现象
gpt-oss 模型(如 gpt-oss-120b)在指令要求「先写解释文本再调用工具」时,会在 commentary channel 生成工具调用的前导说明(preamble),此时 `recipient=None`。解析 `harmony_utils.py` 时崩溃:日志报 `ValueError: Unknown recipient: None`(vllm/entrypoints/harmony_utils.py:438)。影响为:推理 chunk 成功发出后流提前终止,`response.completed` 永远不会发送,客户端看到「Premature end of chunk coded message body: closing chunk expected」及 `curl: (18) transfer closed with outstanding read data remaining`。

## 根因
正文披露根因:按 OpenAI Harmony 格式文档,commentary channel 可以包含「调用多个函数的前导说明」——即不针对特定函数的解释性文本,因此 `recipient=None` 是合法输出。而 `parse_output_message()` 此前只处理有具体 recipient 的 commentary(`functions.X`、`python`、`browser`、`container`),遇到 `recipient=None` 落入 else 分支直接抛 `ValueError: Unknown recipient: None`。

## 修复方案
改 `vllm/entrypoints/harmony_utils.py` 的 `parse_output_message()`:commentary channel 的内置工具分支改为同时接受 `recipient is None`,把无 recipient 的前导文本作为 reasoning 内容解析(语义正确,因为 preamble 是解释性内容);未知 recipient 仍抛 ValueError。

关键 diff 摘录:
```diff
-        elif recipient is not None and (
-            recipient.startswith("python")
-            or recipient.startswith("browser")
-            or recipient.startswith("container")
+        elif (
+            recipient is None  # Preambles: explanatory text before tool calls
+            or recipient.startswith(("python", "browser", "container"))
         ):
             output_items.extend(_parse_reasoning_content(message))
```
另在 `tests/entrypoints/test_harmony_utils.py` 新增 `TestParseOutputMessage` 一组单测,覆盖 recipient=None、functions.X、python/browser/container、空内容、多 content、未知 recipient 报错、analysis channel、非 assistant 消息等场景。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:仅 `vllm/entrypoints/harmony_utils.py` 的 commentary 解析分支;受影响的是 gpt-oss 系模型 `/v1/responses` 接口在带工具+要求解释性文本的指令下的输出解析。修复后 `recipient=None` 的 commentary 从抛异常变为生成 reasoning item;`functions.X` 仍生成 function call;未知 recipient 的报错语义保留(有单测锁定)。CLI/其他 channel 解析不受影响。

## 关联
[#24170](https://github.com/vllm-project/vllm/issues/24170)(正文引用的 issue)、[Harmony format docs](https://cookbook.openai.com/articles/openai-harmony)(正文引用的格式说明)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
