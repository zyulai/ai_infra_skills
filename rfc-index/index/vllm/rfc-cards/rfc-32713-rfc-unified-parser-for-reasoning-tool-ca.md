# RFC #32713: [RFC]: Unified Parser for reasoning, tool calling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32713](https://github.com/vllm-project/vllm/issues/32713) |
| 作者 | qandrew |
| 创建日期 | 2026-01-20 |
| 状态 | ⚪ closed(关闭于 2026-07-24) |
| labels | RFC, stale |
| 热度 | 👍 11 · 💬 12 |
| 分类 | Other |

## 一句话结论
提议统一 Parser 类合并 tokenization、reasoning 与 tool calling 解析入口,长期按模型自动推导并移除相关启动参数。

## 提案动机
LLM 本质是 token 进 token 出,需要文本→token 的 tokenization 与 token→有用响应的解析(detokenization、reasoning 解析、tool calling 解析)。因历史原因,vLLM 的 reasoning parser 与 tool calling parser 入口是分离的(reasoning parser 的需求到 deepseek R1 才出现)。现状问题:OpenAI Harmony 是分叉代码路径需单独开发,DeepSeek v3.2 因不用 jinja 模板又产生一次分叉;tokenization(`_preprocess_chat()`)、tool 解析、reasoning 解析三套入口承载大量相似逻辑。

## 提出的方案
- 单一统一 `Parser` 类:`tokenize()`(现有模型进入 `engine._preprocess_chat()`,GPT-OSS/自定义模型可在此自行 tokenize)、`extract_reasoning()`(委托 reasoning_parser)、`extract_tool_calls()`(委托 tool_parser)
- 兼容两条路径:legacy 流程(jinja 模板 + 调用 reasoning/tool 子类)与自定义实现(DeepSeek v3.2 经 `tokenize()` 进入,GPT-OSS 的编解码也经 Parser 进入),并为未来特性(交错推理、并行工具调用)留扩展位
- 长期移除 `--tool-call-parser`、`--reasoning-parser`、`--enable-auto-tool-choice` 标志,Parser 由所服务模型自动定义,也避免错误组合(如给 deepseek 模型配 Llama reasoning parser)
- 任务清单:已完成 Parser 类初版并在 responsesAPI 落地(PR #32712);待办:推广到 chat completions 等其他入口、支持并行工具调用与交错推理、迁移 GPT-OSS Harmony 以减少分叉、废弃旧参数
- 更新(2/17):本 RFC 只覆盖 decode 侧解析;tokenization 不包含在 Parser 内,改由 renderer 处理

## 优化点与收益
正文未给出量化数字。收益为消除三套入口与 Harmony/DeepSeek 分叉路径、减少配置错误可能。

## 现状与落地
已关闭(2026-07-24,label 含 stale)。首版已在 PR #32712 落地,清单其余任务未勾选;后续落地情况需查关联 PR。

## 关联
- PR #32712:https://github.com/vllm-project/vllm/pull/32712

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
