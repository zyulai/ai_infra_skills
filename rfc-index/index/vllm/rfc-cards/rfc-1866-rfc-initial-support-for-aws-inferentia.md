# RFC #1866: [RFC] Initial Support for AWS Inferentia

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#1866](https://github.com/vllm-project/vllm/issues/1866) |
| 作者 | liangfu |
| 创建日期 | 2023-11-30 |
| 状态 | ⚪ closed(关闭于 2024-03-02) |
| labels | (none) |
| 热度 | 👍 16 · 💬 9 |
| 分类 | Feature / Ecosystem |

## 一句话结论
集成 transformers-neuronx 作为执行引擎,让 vLLM 支持 AWS Inferentia 上的 LLM 推理。

## 提案动机
正文未设独立 Motivation 章节,直接以 Proposal 开头:目标是在 vLLM 中支持 Inferentia 上的 LLM 推理,方案是集成 transformers-neuronx 作为执行引擎,该集成需要对 transformers-neuronx 与 vLLM 两边同时改动。

## 提出的方案
- **transformers-neuronx 侧改动**:支持 batch size 1 的 prompt encoding 并与最大 batch size 的 decoding 共享同一 cache 空间;支持 batch 相关的 KV cache 更新(每个序列用指定的 position_id 更新 cache);支持 virtual dynamic batching,使多 batch prompt encoding 对 vLLM 侧几乎无感。
- **vLLM 侧改动**(正文 checklist 全部勾选):CUDA kernel 编译改为可选(inf2 实例上无需编译 CUDA kernel,默认仍保持开启,PR #2065);将 transformers-neuronx 作为可选第三方依赖(其进一步依赖 torch-neuronx、torch-xla、neuronx-cc 等,PR #2065);在 vLLM model loader 中配置 transformers-neuronx 启用 continuous batching(PR #2569);加载权重后编译模型(PR #2569);用 transformers-neuronx 执行模型(PR #2569)。
- **实现细节**:给出 neuron 专用的模型级 forward 函数(prompt encoding 时按 slot_mapping 推导 seq_ids,decoding 时用 block_tables);模型编译流程(用 `save_pretrained_split` 拆分 HF 权重,再 `LlamaForSampling.from_pretrained` + `to_neuron()`);以及通用 model loader(通过 `NeuronConfig` + `ContinuousBatchingConfig`,传入 tp_degree、amp='f32'、context_length_estimate、batch_size 等参数)。

## 优化点与收益
正文未给出量化数字。收益为生态扩展:使 vLLM 获得在 AWS Inferentia(inf2)实例上运行 LLM 推理的能力,并借助 transformers-neuronx 的 continuous batching 与共享 cache 设计复用 vLLM 的调度能力。

## 现状与落地
已关闭(2024-03-02)。正文清单中 vLLM 侧改动项全部勾选完成,并关联 PR #2065 与 #2569,显示提案内容已通过这两个 PR 落地;细节可再查关联 PR 确认。

## 关联
- PR [#2065](https://github.com/vllm-project/vllm/pull/2065)(CUDA 编译可选 + 可选依赖)
- PR [#2569](https://github.com/vllm-project/vllm/pull/2569)(continuous batching、权重加载后编译、执行模型)
- [transformers-neuronx 安装源](https://pip.repos.neuron.amazonaws.com/transformers-neuronx/)(正文给出的 pip 安装地址)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
