# RFC #30263: [Feature] Per-request opt-out of speculative decoding (RFC: disable_speculative_decoding sampling param)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30263](https://github.com/sgl-project/sglang/issues/30263) |
| 作者 | anencore94 |
| 创建日期 | 2026-07-06 |
| 状态 | ⚪ closed(关闭于 2026-09-05) |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
新增布尔 sampling 参数 disable_speculative_decoding,让单个请求声明跳过投机解码,补上"能测量、不能行动"的缺口。

## 提案动机
draft 模型的接受率依赖 workload:同一部署混合负载时,部分请求从投机解码获益甚少甚至为负。投机解码在负载下并不免费——验证成本随 batch_size × draft_length 增长,低 batch 有益、高 batch 可能降低 goodput(这正是 --speculative-adaptive 存在的原因)。当前 SD 每 server 全有或全无(启动时 --speculative-* 固定),而 SGLang 已在 meta_info 暴露逐请求投机统计(spec_verify_ct、spec_accept_rate、spec_accept_length)——能测量却无旋钮行动。本 RFC 是 Adaptive Speculative Decoding 路线图(#23705)中"按接受长度推断"策略的运维声明式补充,把问题缩到最小具体件:逐请求 opt-out flag + 正确 enforcement 的设计讨论。

## 提出的方案
- API:SamplingParams 增加 disable_speculative_decoding: bool = False,暴露于 CompletionRequest / ChatCompletionRequest 与原生 /generate sampling_params;为 True 时该请求输出仅由 target 模型产生、等价普通解码;server 未配置 SD 时 no-op。
- draft PR #30264 已实现端到端 API plumbing(SamplingParams → OpenAI / /generate → msgpack IPC → Req),含 CPU 单测与文档;刻意不做 worker 侧 enforcement——那才是真正的设计难点。
- enforcement 非平凡:num_steps==0 的 trivial-verify 路径只有在 steps-0 runtime state 激活时才一致(AdaptiveController 经 apply_runtime_state 原子换装 target CUDA-graph runner、attention backend 与 steps/draft_tokens);静态 K>0 状态下走该分支会破坏 num_tokens_per_req == draft_token_num 不变量——CUDA-graph 路径形状不匹配崩溃,eager 路径各后端 attention metadata 互相矛盾。
- 三个选项:(A) 为全 opt-out 的 batch 预建 steps-0 SpecRuntimeState 并应用/恢复(batch 粒度,受 adaptive 支持矩阵门控,预建有额外 capture 时间与显存成本);(B) 调度器把 opt-out 请求聚成同质非 spec batch(长期正解,改动大且与 continuous batching 交互);(C) 作为独立请求类路由(如 PD/disagg 下的非 spec worker),超出本 RFC 范围。作者倾向先 (A)、(B) 跟进。
- 限制:enforcement 初期仅覆盖 EAGLEWorkerV2 verify 路径(EAGLE / EAGLE3 / NEXTN);(A)/(B) 下 draft 的 prefill/extend 仍运行以保温 draft KV;混合 batch 中 flag 被忽略(是否告知客户端为开放问题)。

## 优化点与收益
正文未给出量化数字。收益是混合负载下的 goodput 保护:让已知不会获益的请求跳过投机,避免验证开销反噬吞吐。

## 现状与落地
已关闭(2026-09-05)。draft PR #30264 明确保持 GitHub Draft、在命名/形态与 enforcement 方案定案前不合并;最终落地需查关联 PR。

## 关联
- #23705([Roadmap] Adaptive Speculative Decoding)
- #9319(动态 SD,被 stale bot 关闭)
- vLLM #7569(runtime API 控制 SD,not-planned)
- vLLM #17984(逐序列 SD 控制,not-planned)
- Draft PR #30264(API plumbing + 测试 + 文档)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
