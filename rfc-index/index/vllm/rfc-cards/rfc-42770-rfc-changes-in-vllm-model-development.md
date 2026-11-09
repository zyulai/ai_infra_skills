# RFC #42770: [RFC]: Changes in vLLM Model Development

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42770](https://github.com/vllm-project/vllm/issues/42770) |
| 作者 | WoosukKwon |
| 创建日期 | 2026-05-15 |
| 状态 | 🟢 open |
| labels | RFC, stale |
| 热度 | 👍 41 · 💬 22 |
| 分类 | Other |

## 一句话结论
重塑 vLLM 模型开发范式:移除全图 torch.compile 依赖、按硬件后端分离模型代码、引入显式 Model/ModelConfig 接口。

## 提案动机
过去为做算子融合不愿直接修改模型代码,累积的抽象与编译器 pass 反而让代码难以理解与优化;"一套模型定义适配所有硬件"的理想不现实——不同硬件后端适合不同融合策略,单一抽象使代码复杂且脆弱(一个后端的改动容易破坏另一个);此外 AI 编码 agent 让手写算子融合大幅变易,而 agent 在原始模型代码上工作效果最好。

## 提出的方案
- **移除全图 `torch.compile` 依赖**(仍保留局部算子融合用途),迁移须保持用户 UX 不变、无性能回退。
- **编译器融合 → 手工融合**:在模型代码中直接引入 `add_rms_norm_quant` 这类"大融合算子",覆盖 DeepSeek V3.2 等编译器处理不好的复杂场景。
- **CustomOp / vLLM IR** 仅为遗留模型保留;新模型用融合算子加自定义派发逻辑(如 GB200/GB300 用 FlashInfer 内核,H100/H200 用其他内核)。
- **Piecewise CUDA graph**:采用 SGL 的 "breakable CUDA graph" 方案(#42304 已有原型),并移除 `forward_context` 等为编译器兼容留下的 hack。
- **按硬件厂商分离模型代码**:如 `deepseek_v4/nvidia/{model.py, kernels/, tests/}` 与 `amd/` 各自独立演进,便于单元测试、AI 辅助优化与淘汰旧模型。
- **显式模型接口**:定义 vLLM 自己的 `ModelConfig`(HF config 与 model protocol 的并集)与 `Model` 抽象(`load_weights`/`init_kv_cache`/`prepare_inputs`/`forward` 等),模型自管理状态;实现两个接口即可被完全支持,模型开发者无需再将模型移植为 HF 格式。

## 优化点与收益
正文未给出量化数字。定性收益:在最新硬件上靠手工融合达到 speed-of-light 性能(编译器优化不够快)、去掉 `torch.compile` 带来的启动时间增加与迭代摩擦、`ModelConfig` 消除 spawn 子进程读取模型字段(常被 import 时间主导)的开销、各硬件后端互不破坏。

## 现状与落地
仍在推进。正文列出 In-Progress:#42304(piecewise CUDA graph 原型)、DeepSeek V4 硬件分离初始实现(AMD 部分由 @tjtanaa 负责)、#43224(迁移编译器实现的融合);TODO:移除 DeepSeek V4 的 `@supports_torch_compile`、移植更多模型(Kimi、GLM、Qwen、Minimax、Nemotron 等)、全部移植后弃用 vLLM 的 torch.compile 集成、定义并应用模型接口。labels 含 stale,后续进展需查关联 PR。

## 关联
- [#42304](https://github.com/vllm-project/vllm/issues/42304)(PW CUDA graph 原型)
- [#43224](https://github.com/vllm-project/vllm/issues/43224)(迁移编译器融合)
- [#24384](https://github.com/vllm-project/vllm/issues/24384)(ModelConfig 思路类似)
- [model runner V2 的 model_states/interface.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/interface.py)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
