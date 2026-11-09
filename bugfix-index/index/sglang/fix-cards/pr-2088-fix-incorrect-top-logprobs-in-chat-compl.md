# PR #2088: Fix: incorrect top_logprobs in chat completion

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2088](https://github.com/sgl-project/sglang/pull/2088) |
| 作者 | ajwaitz |
| 合并日期 | 2024-11-19 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
chat completion 的 `top_logprobs` 恒取第 0 步解码结果,所有 token 位置内容相同;循环缺位置索引,改为按 `token_idx` 取对应步的 top logprobs。

## 问题现象
chat completion 请求带 `logprobs=True, top_logprobs=3` 时,返回对象 `logprobs.content` 中每个 `ChatCompletionTokenLogprob` 的 `top_logprobs` 字段完全相同。按 OpenAI 语义,序列中第 n 个 token 应携带第 n 步解码捕获的 top logprobs,实际却是每个元素都填了第 0 步的 top tokens。作者用 `meta-llama/Llama-3.2-1B-Instruct`、prompt "What is the capital of France?"、`temperature=0, max_tokens=128` 复现:8 个输出 token(从 'The' 到 '<|eot_id|>')的 `top_logprobs` 全部是 `[The(-0.0065), Paris(-5.0846), R(-9.5065)]`,即首步解码的候选,与各自的 `logprob`(token 本身概率)明显不匹配。

## 根因
正文披露(Modifications):问题出在 `sglang/srt/openai_api/adapter.py` 的 `v1_chat_generate_response` 组装响应时 logprob 列表的索引方式。生成循环里虽然逐 token 遍历 `logprobs.tokens` 与 `logprobs.token_logprobs`,但读取 top 候选时固定写死 `logprobs.top_logprobs[0]`,未随当前 token 的位置推进,导致每个位置都引用第一步解码的 top-logprob 字典。

## 修复方案
`v1_chat_generate_response` 中给遍历加上 `enumerate`,用当前 token 下标索引 top_logprobs 列表:

```diff
-            for token, logprob in zip(logprobs.tokens, logprobs.token_logprobs):
+            for token_idx, (token, logprob) in enumerate(
+                zip(logprobs.tokens, logprobs.token_logprobs)
+            ):
                 token_bytes = list(token.encode("utf-8"))
                 top_logprobs = []
                 if logprobs.top_logprobs:
-                    for top_token, top_logprob in logprobs.top_logprobs[0].items():
+                    for top_token, top_logprob in logprobs.top_logprobs[
+                        token_idx
+                    ].items():
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围(修复前所有启用 chat completion `top_logprobs` 的版本均受影响,按正文现象如实转述)。
- 回归面:仅 `openai_api/adapter.py` 中 chat completion 响应组装的 `top_logprobs` 字段;`logprobs.content` 各元素此后引用各自解码步的候选。若上游 `top_logprobs` 列表长度与 token 数不一致,新索引可能暴露该不一致(diff 改动的直接后果,正文未讨论)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
