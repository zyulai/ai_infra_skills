# RFC #28326: [RFC]: vLLM Support for Generic Model Definitions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28326](https://github.com/vllm-project/vllm/issues/28326) |
| 作者 | bwasti |
| 创建日期 | 2025-11-07 |
| 状态 | ⚪ closed(关闭于 2026-03-12) |
| labels | RFC, stale |
| 热度 | 👍 15 · 💬 7 |
| 分类 | Feature / Ecosystem |

## 一句话结论
为用户自定义模型定义提供规范与工具,让训练代码可直接接入 vLLM 获得快速推理。

## 提案动机
用户在实验模型架构时,常需要一条快速推理路径来探索 post-training(强化学习)。由于该领域的训练是推理主导的(inference dominated),vLLM 应当通过提供工具、让用户能以低认知开销从训练代码直接构造快速推理模型来支持研究。正文提到:近期基于 vLLM 的 bitwise-exact 强化学习探索(torchtitan 的 deterministic_vllm_rl 实验)表明,利用 vLLM 的特性对保证训练运行稳定至关重要;且 vLLM 已被 VERL、TorchForge 等 RL 框架使用,简化集成有助于扩大 vLLM 的影响力与社区。

## 提出的方案
目标是支持通用的「训练等场景 ⇔ 推理」兼容性:为 `ModelRegistry.register_model` 在训练及其他负载下的使用提供正式的 **specification**(规范)与 **helper utilities**(辅助工具),具体包括:
- 一份规范,定义用户自定义模型使用 vLLM 所需的最小特性集;
- 一个 wrapper,通过 shim 接口或特性,让「仅训练」用途的模型定义能以简单方式接入 `register_model`;
- 一组 vLLM 官方提供的 `nn.Module`(如 `Attention`),同时支持训练与高效推理;
- 保持灵活性以支持用户自定义并行,具体是把 process group 的实例化暴露给 model registration。

## 优化点与收益
正文未给出量化数字。收益面向研究与生态:降低训练代码接入 vLLM 推理的门槛,服务于 RL post-training 场景的快速迭代,并与现有 RL 框架(VERL、TorchForge)的集成路径互补。

## 现状与落地
已关闭(2026-03-12),labels 含 stale;正文未写明是否已实现或被替代,落地情况需查关联 PR。

## 关联
- [vLLM 自定义模型文档](https://docs.vllm.ai/en/latest/contributing/model/basic/)(正文称其为「目前非常接近、但需要侵入式改动」的接口)
- [vllm/model_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)(module 样板代码所在)
- [vllm/model_executor/layers](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers)(正文建议在此添加 attention 的反向传播实现)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
