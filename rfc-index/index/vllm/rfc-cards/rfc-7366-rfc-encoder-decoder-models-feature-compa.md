# RFC #7366: [RFC]: Encoder/decoder models & feature compatibility

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7366](https://github.com/vllm-project/vllm/issues/7366) |
| 作者 | afeldman-nm |
| 创建日期 | 2024-08-09 |
| 状态 | ⚪ closed (2025-09-18) |
| labels | RFC, stale |
| 热度 | 👍 14 · 💬 25 |
| 分类 | Other |

> 注:本卡所用 pack 的正文在约 20k 字符处截断("Support custom attention bias"一节中途截断),卡片基于现有内容撰写。

## 一句话结论
encoder/decoder 功能成熟度路线图:用支持矩阵列出待补齐的模型(Whisper、T5)与特性(量化、多模态、注意力后端等)。

## 提案动机
社区对 vLLM 支持 encoder/decoder 模型有浓厚兴趣(issue #187、#180 即为请求),该能力已通过三个 PR(#4837、#4888、#4942)引入。但这三个 PR 只做到了"能推理",在两方面仍有欠缺:(1) encoder/decoder 请求处理管线与 decoder-only 管线在特性支持上的差距;(2) 受支持的 encoder/decoder 模型数量太少。作者呼吁社区提交 PR,把 encoder/decoder 功能提升到与 decoder-only 功能相当的成熟度。

## 提出的方案
以"支持矩阵 + 分工作流详述"组织:
- 支持矩阵:已具备——encoder/decoder 基础设施、BART;长期目标——Whisper、T5 及其他 enc/dec 模型、量化(当前未测试)、多模态、Xformers 之外的注意力后端(尤其是 flash-attn、flashinfer)、自定义 attention bias、CUDAGraph(issue #7447)、pipeline parallelism;投机解码与前缀缓存为低优先级且困难;sliding window、chunked prefill、LoRA 明确不是长期目标;
- 加 Whisper:扩展多模态支持到 enc/dec、prompt 管线支持音频、移植 HF Whisper 模型(已有 PR #5964)、各层支持 TP>1、补测试;可考虑与多模态支持同 PR 以便端到端验证;
- 加 T5:依赖至少一个后端同时支持 encoder/cross attention 与自定义 attention bias(T5 相对位置编码所需),当时无后端满足;移植 HF 模型并支持 TP>1,旧版本 vLLM 的 #3117 可作起点;
- 量化:为所有 decoder-only 已支持的量化方法补 enc/dec 单测;作者预期大部分量化方法天然兼容(量化 kernel 只作用于权重矩阵 GEMM,而 enc/dec 只改 `Attention` 层行为),FP8 是否兼容存疑(其使用专用量化 KV cache kernel);
- 多模态:Llava 类模型不走 cross-attention 故不受影响,但 Whisper 这类 cross-attention 模型完全无多模态支持(`EncoderDecoderModelRunner` 有 assert);初步提出带 `multi_modal_data` 的 `TextPrompt`/`TokensPrompt` 两种格式——多模态数据传给 encoder、文本/ token 传给 decoder;
- 注意力后端扩展:当时仅 XFormers 支持 encoder/cross attention;列出后端需改造的要点(`AttentionMetadata` 编码器字段、`attn_type` 参数、causal/非 causal 及 mask 形状选择),初期目标是 flash-attention 与 flashinfer,并移除强制 XFormers 的逻辑;长期目标是所有后端支持;
- 自定义 attention bias:允许在 softmax 前向 SDP 分数矩阵加任意矩阵 A(现有后端只支持均匀 0/−inf 的块对角 mask),T5 依赖此能力(该节正文在截断处中断)。

## 优化点与收益
收益在于把 encoder/decoder 管线的特性覆盖(量化、多模态、高性能注意力后端、CUDA Graph 等)追平 decoder-only 管线,扩大可服务模型面(Whisper 语音、T5 等)。正文未给出量化数字。

## 现状与落地
已关闭(2025-09-18),labels 含 stale。正文列出的是长期目标与工作流步骤,末尾的 "Initial goal" 是号召社区认领矩阵中的模型/特性写 PR;issue 本身未记录完成情况,落地情况需查关联 PR。

## 关联
- https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/infra-enc-dec.md (enc/dec 基础设施详解)
- https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/how-to.md (新增模型/特性指南)
- https://github.com/vllm-project/vllm/pull/5964 (Whisper 移植 PR)
- https://github.com/vllm-project/vllm/issues/7447 (enc/dec CUDA Graph issue)
- https://github.com/vllm-project/vllm/issues/3117 (旧版 T5 实现,可作起点)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
