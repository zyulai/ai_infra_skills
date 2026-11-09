# RFC #7692: [RFC]: Add Ascend NPU as a new backend

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7692](https://github.com/vllm-project/vllm/issues/7692) |
| 作者 | wangshuai09 |
| 创建日期 | 2024-08-20 |
| 状态 | ⚪ closed (关闭于 2025-02-21) |
| labels | RFC |
| 热度 | 👍 39 · 💬 2 |
| 分类 | Hardware / Backend |

## 一句话结论
按 vLLM 既有后端接入机制新增 Ascend NPU 后端,让用户可以在华为昇腾 NPU 上运行 vLLM。

## 提案动机
vLLM 提供易用的后端接入机制,已集成多种后端;如 issues #6368、#6728、#6066 所示,许多用户希望在 Ascend NPU 上使用 vLLM。RFC 的主要目的是遵循现有后端接入机制,使 Ascend NPU 可用于 vLLM。Ascend NPU 是一系列采用 NPU 架构的 AI 处理器,onnxruntime、deepspeed、llama.cpp 等项目已提供支持。

## 提出的方案
- 基于 `GPU Executor/Worker(s)` 引入 `Ascend Executor/Worker(s)`,负责 Ascend 运行时管理与 worker
- 以 `Ascend Backend` 替换 attention 层,Page Attention / Flash Attention 算子在该后端实现
- torch_npu 自 2.1.0 起原生支持 torch,因此尽量与 GPU 代码保持一致,实现中做最少的代码改动
- 初始版本包含:Ascend Executor、Ascend Worker、Ascend Model Runner、Ascend MindIE Backend(MindIE 为昇腾推理引擎)、Ascend SingleOps Backend
- 反馈期一个月,CC @mgoin、@WoosukKwon

## 优化点与收益
正文未给出量化数字。定性收益:遵循既有后端接入机制、与 GPU 代码路径保持一致从而减少维护成本,扩展 vLLM 的硬件覆盖面。

## 现状与落地
已关闭。正文为提案与初始 RoadMap,未写明实现结论,落地情况需查关联 PR。

## 关联
- #6368
- #6728
- #6066
- MindIE:https://www.hiascend.com/en/software/mindie

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
