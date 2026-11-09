# RFC #8967: [RFC]: Reward Modelling in OpenAI Compatible Server

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#8967](https://github.com/vllm-project/vllm/issues/8967) |
| 作者 | noamgat |
| 创建日期 | 2024-09-30 |
| 状态 | ⚪ closed(关闭于 2024-11-01) |
| labels | RFC |
| 热度 | 👍 7 · 💬 30 |
| 分类 | Frontend / Serving / API |

## 一句话结论
提议在 OpenAI 兼容服务器中优雅支持 reward model,列出四种方案,作者倾向为 EmbeddingRequest 增加对话输入。

## 提案动机
Reward model 是 NLP/AI 工作流的重要工具,尤其用于 agentic 流程中验证中间输出质量、或在同一任务的多次尝试之间排序。vLLM 刚通过 PR #8896 加入 reward model 支持,但在 OpenAI 兼容服务器上只能「搭载」embedding 端点使用:客户端需知晓服务端所用 tokenizer、自行套用 chat template,再把拼好的字符串发给 embedding 端点,破坏了客户端与服务端的解耦。根因是现有端点都不匹配:embedding 端点接收字符串而非对话;chat 端点返回字符串而非数值,即使请求 logprobs 也已经过 softmax、不可逆。

## 提出的方案
- Option 1(作者支持,认为最少 hack 且新代码量小):给 `EmbeddingRequest` 增加 conversation 对象(`List[Dict[str, str]]`)输入;服务端检测到后用 tokenizer 应用 chat template,再按 `str` 输入继续处理
- Option 2:让 chat 端点可返回 logits 而非 logprobs——新增 per-request 参数(类似 `top_logprobs`),或服务端标志 `--return_logits_instead_of_logprobs`
- Option 3:为 vLLM 增加专用端点
- Option 4:不改动——改用已接受对话输入的 `/tokenize` 端点,先取 token 列表再调 embeddings,同样解决耦合问题

## 优化点与收益
正文未给出量化数字。收益为工程层面的解耦:客户端不再需要了解服务端 tokenizer 与模板细节。

## 现状与落地
已关闭(2024-11-01)。正文以四个选项征求意见(作者预计讨论期不超过几周),未记录最终选择与实现情况;落地情况需查关联 PR #8896。

## 关联
- PR #8896 相关讨论:https://github.com/vllm-project/vllm/pull/8896#issuecomment-2381912003

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
