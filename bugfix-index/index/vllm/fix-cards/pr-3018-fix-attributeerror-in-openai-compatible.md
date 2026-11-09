# PR #3018: Fix: `AttributeError` in OpenAI-compatible server

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3018](https://github.com/vllm-project/vllm/pull/3018) |
| 作者 | jaywonchung |
| 合并日期 | 2024-02-29 |
| labels | (none) |
| 热度 | 👍 6 · 💬 4 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
OpenAI 兼容服务器在 `add_generation_prompt=False` 时因 messages 属性访问笔误抛 `AttributeError`,改为按字典取键并收紧类型注解。

## 问题现象
正文描述:当请求的 `add_generation_prompt` 为 `False` 时,OpenAI 兼容服务器的 chat 补全路径抛出 `AttributeError`。正文仅附一张报错截图(未提供文字版),未给出日志细节与影响面;触发条件即 chat 请求关闭 `add_generation_prompt`,代码走 `get_chat_request_role` 的 else 分支访问 `messages[-1].role`。

## 根因
正文披露:这是一个小笔误(small typo)。`ChatCompletionRequest.messages` 实际是字典列表,`serving_chat.py` 的 `get_chat_request_role` 却按对象属性访问写成 `request.messages[-1].role`,`add_generation_prompt=False` 时执行该行即触发 `AttributeError`。

正文 side note(作者提出的疑问,非定论):`messages` 的类型注解 `Union[str, List[Dict[str, str]]]` 与 OpenAI chat API 规范不符,规范要求它是数组;若 `messages` 真可能是 `str`,修改后的代码行仍会报错、需要另行处理该情况。作者称对 vLLM 核心 schema 不够熟悉,本 PR 采取把注解收紧为 `List[Dict[str, str]]` 的最小修复(由 diff 推断:收紧注解是为消除 str 分支、保证新写法的类型正确)。

## 修复方案
改动两个文件:
- `vllm/entrypoints/openai/serving_chat.py` 的 `get_chat_request_role`:else 分支由属性访问改为字典取键;
- `vllm/entrypoints/openai/protocol.py` 的 `ChatCompletionRequest.messages` 类型注解由 `Union[str, List[Dict[str, str]]]` 改为 `List[Dict[str, str]]`。

关键 diff(有效改动共 2 行):
```diff
-            return request.messages[-1].role
+            return request.messages[-1]["role"]
```

## 影响与回归风险
- 回归面:按 diff 实际改动,仅涉及 OpenAI 兼容 chat 请求的两处:`get_chat_request_role` 的非 `add_generation_prompt` 路径,以及 `ChatCompletionRequest` 的 Pydantic 类型校验。若确有客户端把 `messages` 传成字符串,收紧注解后该请求会在校验阶段被拒(正文 side note 已提示此风险,并称届时需再处理)。

## 关联
正文未列出 issue/PR 链接(仅附一张截图)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
