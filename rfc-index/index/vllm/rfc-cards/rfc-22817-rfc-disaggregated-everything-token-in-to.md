# RFC #22817: [RFC]: Disaggregated Everything - Token In <> Token Out API Server

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22817](https://github.com/vllm-project/vllm/issues/22817) |
| 作者 | robertgshaw2-redhat |
| 创建日期 | 2025-08-13 |
| 状态 | 🟢 open |
| labels | RFC, keep-open, rl |
| 热度 | 👍 17 · 💬 9 |
| 分类 | Disaggregated Serving |

## 一句话结论
把 Renderer 抽到 API server 层,让 AsyncLLM 变成 token 进/token 出,对外暴露标准 token 级 API 并支持 tokenize 分离部署。

## 提案动机
多个团队向 vLLM 提出提供 token-in / token-out API 的需求,包括:分离式 tokenization / 多模态输入处理场景(如 llm-d、Dynamo 等)、围绕 vLLM 的自定义前端(用户做自定义工具调用、MM 输入处理)、以及 RL 用例。当前 vLLM 引擎内部承担了 OpenAI 请求解析、多模态处理、detokenization 等职责,外部无法在这些环节之前或之后接管。

## 提出的方案
- **Renderer 重构基础**:依托现有的 MM 输入处理重构(#22880),把 `Processor` 类转为 `Renderer` 类——抽象层数更少、不必依赖 HF Processor(允许模型提供方自定义实现)、且位于 AsyncLLM 之外(API server 层)。
- **Renderer 职责**:将 OpenAI Spec 转换为 Tokens、MM Features 等;AsyncLLM 由此变为 tokens in >> tokens out。
- **Token-In / Token-Out API**:定义 `GenerateRequest`(request_id、token_ids、features、placeholder_range、mm_identifier、sampling_params、kv_transfer_params 等,大致映射 `EngineCoreRequest`)与 `GenerateResponse`/`GenerateResponseStreaming`(raw tokens、logprobs、prompt_logprobs、finish_reason、stop_reason、kv_transfer_params,大致映射 `EngineCoreOutput`);提供 `/generate` 与 `/generate_streaming` 端点,以 `vllm serve-tokens $MODEL` 启动 token 微服务;`AsyncLLM` 中的 Detokenization 步骤可设为可选。
- **分离式 tokenization 架构**:两个附加微服务——`vllm renderer $MODEL`(`/v1/chat/completions/render`,输入 ChatCompletionRequest、输出 GenerateRequest)与 `vllm coordinate $MODEL`(`/v1/chat/completions/coordinate`,负责送入 vLLM、把 GenerateResponse 转回 ChatCompletionResponse、detokenization、tools 等)。

## 优化点与收益
正文未给出量化数字。收益为架构层面:token 级 API 使 tokenization/多模态处理/工具调用可从引擎中拆出独立部署,服务于大规模 serving 的分离式 tokenization、自定义前端与 RL 场景。

## 现状与落地
仍在讨论/待实现:状态 open(labels 含 keep-open),Feedback Period 一栏为空,方案以提案形式给出,尚待社区评审。

## 关联
- #22880(MM 输入处理重构 PR,Processor → Renderer)
- [Architecture Diagrams](https://docs.google.com/drawings/d/1MITogNEEpRAUQ3TkjdcCprK1jrQyORsuhVbNWeZpLSg/edit)(正文附带的架构图)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
