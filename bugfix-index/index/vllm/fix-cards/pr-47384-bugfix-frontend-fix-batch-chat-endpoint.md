# PR #47384: [Bugfix][Frontend] Fix batch chat endpoint corrupting logprobs when return_token_ids is set

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47384](https://github.com/vllm-project/vllm/pull/47384) |
| 作者 | fenghourun |
| 合并日期 | 2026-07-03 |
| labels | bug, frontend, ready |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
batch chat 端点把 logprobs 渲染错接到 return_token_ids,静默把 token 变成占位符;补字段并重新接线修复。

## 问题现象
`/v1/chat/completions/batch` 端点把 logprobs 渲染接到了 `request.return_token_ids` 而不是 `return_tokens_as_token_ids`。只要设置 `return_token_ids`(本应只额外添加单独的 `token_ids` 列表),每个 logprob 的 `token` 字段就会被静默渲染成 `"token_id:{id}"` 占位符,破坏 logprobs 输出。

## 根因
正文披露:`batch_serving.py` 的 logprobs 渲染调用用错了请求字段(`return_as_token_id=request.return_token_ids`);同时 `BatchChatCompletionRequest` 完全缺失 `return_tokens_as_token_ids` 字段(非 batch 的 `ChatCompletionRequest` 上有),导致 batch 请求无从正确指定该行为。作者检索过 open/closed PR 与 issue,确认无重复修复。

## 修复方案
- `vllm/entrypoints/openai/chat_completion/protocol.py`:`BatchChatCompletionRequest` 新增 `return_tokens_as_token_ids: bool | None = None`,默认 `None` 以回落到服务端 `--return-tokens-as-token-ids`,与非 batch 请求一致;
- `vllm/entrypoints/openai/chat_completion/batch_serving.py`:logprobs 调用改接正确字段:

```python
-                        return_as_token_id=request.return_token_ids,
+                        return_as_token_id=request.return_tokens_as_token_ids,
```

`return_token_ids` 的正确用法(token-ID 列表)不受影响。新增 GPU-free 单测(字段存在、默认 None)与两条 `RemoteOpenAIServer` 集成回归测试(仅设 `return_token_ids` 时不出现占位符;显式设 `return_tokens_as_token_ids` 时出现占位符)。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅 OpenAI 前端 batch chat 端点的 logprobs 渲染与请求协议;非 batch 端点与采样/引擎路径无改动。两条集成测试需要 GPU,本地无 GPU,依赖 CI 验证(正文如实注明);`ruff-check`/`ruff-format` 通过。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
