# RFC #47161: [RFC]: Streaming Derender for Disaggregated Serving

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47161](https://github.com/vllm-project/vllm/issues/47161) |
| 作者 | hickeyma |
| 创建日期 | 2026-06-30 |
| 状态 | ⚪ closed(关闭于 2026-10-22) |
| labels | RFC, kv-connector, tool-calling |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Disaggregated Serving |

## 一句话结论
为 derender 后处理端点增加流式支持,让 P/D 分离下的交互式消费方逐 chunk 完成去分词与工具/推理解析。

## 提案动机
已有的批量 derender 端点(`/v1/chat/completions/derender`、`/v1/completions/derender`,来自 #43606 去分词与 #45919 非流式解析)只消费完整 `GenerateResponse`、整段返回 `ChatCompletionResponse`。但 llm-d、Dynamo 等交互式 serving 场景要求 token 边生成边返回客户端,流式是硬性门槛;而生成端 `/inference/v1/generate` 已支持 SSE 流式输出,derender 侧需要对应的逆路径。

## 提出的方案
- 复用两个 derender 路径加 `stream: true` 标志,FastAPI 以 typed union 按 `stream` 字面量判别、handler 用 `isinstance` 分发;流式输入是 `GenerateStreamResponse` 增量,与批量路径的完整响应是不同请求形态,故不是加参数而是新形态。
- 核心设计 Option A(无状态):服务端每响应携带 `stream_state`,客户端下一 chunk 原样回传,服务端无 session、无生命周期管理;拒绝 Option B(`request_id` 键控服务端会话),Option C(shm 状态传输)延后。
- 解析器状态用 replay 而非序列化:每个 chunk 重建全新 parser,按 `output_chunk_lens` 记录的原始 chunk 边界逐 chunk 重放 `parse_delta`(每 chunk 一次而非每 token),保证多 token step(如投机解码)重建出相同 parser 状态;工具调用 ID 在状态中固定,重放不改变客户端已见过的 ID。
- 去分词状态直接携带增量 offset:`prev_tokens`(尾部窗口)、`prefix_offset`、`read_offset`,计算 O(delta)/chunk、传输 O(window),状态不随生成长度增长;CJK/emoji 逐 token parity 测试守护 #46159 不回归。
- 响应为每调用一个 JSON `{chunk, stream_state}`,明确拒绝 SSE(与无状态设计冲突);客户端可控的 `stream_state` 做 fail-closed 校验(缺 `chat_request`/`prompt_token_ids`、单 chunk 多 choice、超出 `max_model_len`、`output_chunk_lens` 不一致等一律 400 不 500)。

## 优化点与收益
正文未给出正向量化收益,但如实给出实测成本:@shimib 的 benchmark(#56851 评论)在同等负载下测得 replay 约为进程内 CPU 的 ~9×,数千 token reasoning 输出 E2E p50 +15%;默认单 worker 时该层在并发 parsed streams 下饱和。重放成本为 O(n²) 调用、对重扫文本的 parser 为 O(n³) 字符工作量;低延迟交互流量改用 #56851(parser 在 decode pool 存活)而非给 render 层加服务端状态。

## 现状与落地
已关闭(关闭于 2026-10-22),主体已实现:PR1 纯 detok 流式 derender #47301、PR2 chat replay 解析路径 #50550、PR3 parity CI + 文档均标记 merged;流式 logprobs #55029 与 plain path 修复 #57528 也已 merged。parser cache(#57571)、共享 per-choice `ChatStreamProcessor`(#57572)等延后另行跟踪。

## 关联
- [#42729](https://github.com/vllm-project/vllm/issues/42729)(批量 derender RFC,本 RFC 的父 issue)
- #47301(PR1,merged)
- #50550(PR2,merged)
- #56851(低延迟替代方案)
- #32713(解析器状态序列化,已关闭为 not planned)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
