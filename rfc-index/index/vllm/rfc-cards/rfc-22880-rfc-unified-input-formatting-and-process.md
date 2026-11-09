# RFC #22880: [RFC]: Unified Input Formatting and Processing via `Renderer`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22880](https://github.com/vllm-project/vllm/issues/22880) |
| 作者 | ywang96 |
| 创建日期 | 2025-08-14 |
| 状态 | ⚪ closed (关闭于 2026-02-18) |
| labels | RFC, keep-open, multi-modality |
| 热度 | 👍 20 · 💬 9 |
| 分类 | Other |

## 一句话结论
用新组件 Renderer 统一 tokenizer、chat template 与多模态输入处理,把"OpenAI 风格请求"直接变成 token ids 与多模态特征。

## 提案动机
vLLM 现有输入处理管线复杂而碎片化:tokenization、chat template 格式化、多模态输入处理分散在 `Processor`、`InputPreprocessor`、`MultiModalContentParser` 等多个组件中。在线 serving 时,prompt 格式化、媒体拉取与 tokenization 在 API server 层,而多模态输入处理在 `AsyncLLM` 内部,需要处理各种输入类型组合,引入不必要的复杂度。此外处理逻辑与 Hugging Face `tokenizers`/`transformers` 紧耦合,模型开发者难以在 vLLM 上支持自定义模型。

## 提出的方案
- 提出新组件 `Renderer`(命名灵感来自 openai/harmony),统一 `Tokenizer`、`Processor` 与 `InputPreprocessor`,职责是把高层 API 请求(OpenAI 风格 JSON spec)转换为 token ids、多模态特征及相关 metadata,供 `EngineCore` 直接消费
- 职责分离:Renderer 负责「specs to tokens and features」(默认位于 API server 层,也可作为独立远程进程),`AsyncLLM` 负责「tokens-in-tokens-out」
- 接口含 `render_conversation`(chat/completion 请求转 token ids + 多模态特征)与 `render_prompt`(`.generate` 端点用);实现不必基于 HF Transformers,甚至不必是 Python,只要符合接口
- 核心数据类 `MultiModalFeatureSpec`:封装 `data`(如 pixel_values)、`modality`、`mm_identifier`(mm_hash/uuid)、`mm_position`(PlaceholderRange);一个多模态条目对应一个 Spec
- 多模态缓存:P0 只需在 LRU map 中存 `mm_identifier` 字符串,与 P1 输入缓存键一致;支持 EPD Disaggregation(可把多模态编码器放进 Renderer 远程处理为 embeddings)
- 反馈期未设定(_No response_);由 @WoosukKwon @DarkLight1337 @huachenheli 与作者联合提案

## 优化点与收益
正文未给出量化数字。定性收益:减少抽象层与碎片化、输入准备集中一处、模型厂商可插拔非 HF 的自定义输入处理实现、Renderer 可独立远程部署。

## 现状与落地
已关闭。正文为联合提案与接口设计(含 Q&A),未写明实现结论,落地情况需查关联 PR。

## 关联
- openai/harmony:https://github.com/openai/harmony
- #22817( disaggregated input processing)
- #22198(多模态输入缓存)
- EPD Disaggregation 论文:https://arxiv.org/abs/2501.05460

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
