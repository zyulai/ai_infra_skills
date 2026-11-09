# RFC #42729: [RFC]: Derender Endpoints to Provide Detokenization for Disaggregated Serving

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42729](https://github.com/vllm-project/vllm/issues/42729) |
| 作者 | hickeyma |
| 创建日期 | 2026-05-15 |
| 状态 | ⚪ closed(关闭于 2026-10-23) |
| labels | RFC, kv-connector |
| 热度 | 👍 5 · 💬 19 |
| 分类 | Disaggregated Serving |

## 一句话结论
在 render server 上新增 /derender 端点做逆渲染:token 输出还原为 OpenAI 响应(含 reasoning/tool 解析)。

## 提案动机
`/render` 端点(`POST /v1/chat/completions/render` 与 `POST /v1/completions/render`,PR #36166)已作为分离式 serving 的预处理环节落地:接受标准 OpenAI 请求,返回 GPU 推理层无需 tokenizer 即可直接消费的 `GenerateRequest`。生成后,推理服务器(`/inference/v1/generate`)返回带 raw output token IDs 的 `GenerateResponse`,其 logprob 条目以 `"token_id:N"` 占位字符串编码(生成服务器没有 tokenizer)。要转成客户端可见的 OpenAI 响应需:1) 把输出 token IDs 反解码为文本;2) 把 `"token_id:N"` logprob 占位符解析为真实 token 字符串;3) 填充 logprob 条目的 `bytes` 字段(OpenAI spec 要求,无 tokenizer 的生成服务器无法填写);4) 格式化为 `ChatCompletionResponse` 或 `CompletionResponse`。当前这一后处理耦合在 GPU server 的 `OpenAIServingChat`/`OpenAIServingCompletion` 内,运行 llm-d、Dynamo 等分离部署的运营者要么在 CPU 层跑完整 vLLM 栈,要么自行实现后处理。

## 提出的方案
- **核心思路**:在现有 render server 上暴露 `/derender` 端点,作为 `/render` 的直接逆操作;并采纳 RFC #34407 对 postprocessing 的完整定义——真正的逆操作必须输出 `reasoning_content` 与 `tool_calls` 而不只是 `content`,因此 reasoning/tool call 解析属于本 RFC 范围,构建于 RFC #32713 的统一 `Parser` 之上(`Parser.parse()` 用于批处理、`Parser.parse_delta()` 用于流式,与 `/v1/chat/completions` 同一套调用)。
- **新端点**:`POST /v1/chat/completions/derender`(DerenderChatRequest → ChatCompletionResponse)与 `POST /v1/completions/derender`(DerenderCompletionRequest → CompletionResponse);运行于现有 render server(`vllm launch render`,PR #36166),无需新进程或新二进制;GPU-free,只需 Renderer(tokenizer);同步、无状态、幂等(调用方已持有完整 `GenerateResponse`,无需流式)。
- **协议新增**:在 `vllm/entrypoints/scale_out/token_in_token_out/protocol.py` 增加两个请求类型,字段含 `generate_response(s)`、`prompt_tokens`/`prompt_token_ids`(由调用方提供,因为 `GenerateResponse` 不携带 prompt token 数;从 `/render` 步骤传 `len(token_ids)` 无额外成本)、可选 `chat_request`/`completion_request`(携带 tools、tool_choice、include_reasoning、chat_template_kwargs 等解析器所需字段;配置了 reasoning/tool parser 时缺 `chat_request` 返回 400 而不是把 parser 标记泄漏进 `content`)。
- **实现细节**:chat 路径 `ServingDerender.derender_chat_response` → `OnlineDerenderer.derender_chat`——校验 model 与资源上限、拒绝空 `token_ids`、解析 logprob 占位符、从 prompt 尾部播种增量解码(#59046)、有 parser 时以 `skip_special_tokens=False` 解码并运行 `Parser.parse()` 得到 reasoning/content/tool_calls,最后构建带 `usage` 的响应;CPU 工作(反解码、logprob 解析、解析)在 renderer 的 executor(`renderer_num_workers`)上运行、不占事件循环(#49396)。completions 路径只做纯反解码,不跑 reasoning/tool 解析。
- **分阶段落地**:detokenization 端点先行(#43606),协议即刻做到 parsing-ready 以便后续无破坏性 API 变更;reasoning/tool call 解析(#45045 被 #45919 取代落地);流式 derender + 流式解析由伴随 RFC #47161 规范(detokenization 落地于 #47301,reasoning/tool 解析落地于 #50550)。定位为 #34407 与 #22817 的部分实现。

## 优化点与收益
正文未给出量化数字。收益为架构层面:llm-d、Dynamo 等分离部署无需在 CPU 层运行完整 vLLM 栈或自行实现后处理,render server 即可完成 token → OpenAI 响应的完整还原(detokenization、logprob 占位符解析、`bytes` 字段填充、reasoning/tool call 解析),并配套 round-trip parity CI(#48617 等)保证与 chat serving 输出一致。

## 现状与落地
已关闭(2026-10-23);正文结尾状态表明确「All items are done」:批处理路径(detokenization #43606、reasoning/tool 解析 #45919、parity CI 与文档)与流式路径(#47161 下的 #47301、#50550)均已完成,后续 parity 修复与共享装配(#63148)也已落地;gRPC/msgpack 传输、torch-free 进程内 parser 库等列为 out of scope。

## 关联
- [#34407](https://github.com/vllm-project/vllm/issues/34407)(GPU-less 前端部署中 postprocessing 的定义来源)
- [#22817](https://github.com/vllm-project/vllm/issues/22817)(Token In/Out API Server,部分实现的基础)
- [#32713](https://github.com/vllm-project/vllm/issues/32713)(统一 Parser)
- [#47161](https://github.com/vllm-project/vllm/issues/47161)(流式 derender 伴随 RFC)
- PR [#36166](https://github.com/vllm-project/vllm/pull/36166)(/render 端点,derender 的宿主)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
