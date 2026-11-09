# RFC #4194: [RFC]: Multi-modality Support on vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#4194](https://github.com/vllm-project/vllm/issues/4194) |
| 作者 | ywang96 |
| 创建日期 | 2024-04-19 |
| 状态 | 🟢 open |
| labels | RFC, keep-open, multi-modality, quantization |
| 热度 | 👍 56 · 💬 98 |
| 分类 | Feature / Ecosystem |

## 一句话结论
多模态支持长期路线图:重构输入处理与配置,让 VLM 端到端接入更简单,并持续优化性能。

## 提案动机
初版多模态设计存在几处问题:`VisionLanguageConfig`/`MultiModalData` 仅支持 `pixel_values` 或 `image_features`,难以覆盖预处理更复杂的模型(如 LLaVA-1.6、Fuyu);`image_feature_size`、`image_token_id` 等参数的开销被推给用户,而这些本可从模型与 processor 配置推断;缺少 processor 工具,阻塞了 OpenAI API 服务器的端到端推理;prompt 格式 `"<Image>" * 576 + prompt` 与 HuggingFace 格式不一致,造成使用困惑。

## 提出的方案
- 新增 processor 工具:像 tokenizer 一样在 `LLMEngine` 中挂载 transformers 的 `AutoProcessor`,让 API 服务器与 `LLM` 对象都能端到端推理。
- 前端输入格式与 HuggingFace 保持一致,预处理隐藏在 `LLMEngine.add_request()` 内部,用户层只需传入图像。
- 重构 `MultiModalData`:仅持有多模态数据字典,在模型 forward 时解包,做法接近 transformers 的多模态模型实现。
- 简化 `VisionLanguageConfig`:动态 shape 时可选 `max_feature_size`,供引擎做最坏情况 profiling 或中止请求。
- 输入类型改为传递多模态 embeddings(CLIP→Projector 输出)而非 image features,更通用。
- 后续里程碑(按更新推进):多模态模型 V1 引擎重构、多模态 prefix caching、输入与 embedding 缓存、混合/交错模态推理、LMM 量化与 LoRA、OpenAI 兼容服务器增强。

## 优化点与收益
正文未给出量化数字。定性收益:启动服务器不再需要指定图像相关参数(从模型仓库推断配置)、支持动态图像 shape、优化 `CLIPVisionModel`/`CLIPAttention`/`MultiModalProjector`,并让多模态与 chunked prefill、prefix caching、流水线并行等性能特性兼容。

## 现状与落地
仍在推进,属活跃的跟踪 issue。正文多次更新:2024-06-11 完成第 1 个重构里程碑,7/3 完成第 2 个,9/8 已完成大部分重构;11/18 更新称未来数月将随 V1 引擎重构聚焦多模态性能优化。P0 项(核心重构、prefix caching、输入/embedding 缓存)基本全部勾选完成;P1/P2 中 LMM 量化、LoRA、OpenAI 兼容服务器增强等部分完成、部分仍开放。

## 关联
- PR [#3042](https://github.com/vllm-project/vllm/pull/3042)(初版多模态支持)
- PR [#3978](https://github.com/vllm-project/vllm/pull/3978)
- [V1 多模态重构设计 Doc](https://docs.google.com/document/d/11_DFQTku6C2aV6ghK21P76ST6uAUVjMlEjs54prtb_g/edit?usp=sharing)
- [实现新 VLM 指南](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
