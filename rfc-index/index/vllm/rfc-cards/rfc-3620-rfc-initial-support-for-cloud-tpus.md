# RFC #3620: [RFC] Initial Support for Cloud TPUs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3620](https://github.com/vllm-project/vllm/issues/3620) |
| 作者 | WoosukKwon |
| 创建日期 | 2024-03-25 |
| 状态 | ⚪ closed (关闭于 2025-03-11) |
| labels | RFC, stale, tpu |
| 热度 | 👍 36 · 💬 17 |
| 分类 | Feature / Ecosystem |

## 一句话结论
为 vLLM 增加基于 PyTorch XLA 的 Google Cloud TPU 后端,支持在线/离线推理与单/多主机张量并行。

## 提案动机
目标是让 vLLM 与 Google cloud TPU 无缝集成,用户可以方便地在 TPU 上进行在线与离线推理。聚焦最新一代 TPU(v4、v5e、v5p),先支持 Gemma 等稠密模型(bfloat16),再扩展到 Mixtral 等 MoE 模型。投机解码、GPTQ/AWQ 量化、多 LoRA serving 暂不在初始范围内。

## 提出的方案
- 新增 TPU executor 与 TPU worker 作为 GPU executor/worker 的对应物;因 GPU 与 TPU 差异显著走独立代码路径,但共享设备无关的组件(scheduler、KV cache manager、tokenizer)
- 运行时先用 PyTorch XLA(JAX 作为未来方向);用 `torch.compile`(openxla_eval backend)将 PyTorch 模型追踪并降级为 XLA graph
- 张量并行时一个 TPU chip 一个 TPU worker,用 Ray 跨 TPU VM 管理_worker;支持同一 TPU pod slice 内多主机推理,暂不做 multi-slice;Megatron 风格模型分区(替换 `nn.Linear` 为 Row/ColumnParallelLinear)
- 每个 op 提供两套实现:`_forward`(可被 XLA/torch.compile 优化)与 `_forward_cuda`(CUDA 自定义算子),运行时按硬件选择;FlashAttention/PagedAttention 无法由 XLA 生成,改用 Pallas kernels
- 动态 shape 处理:decode 按 batch_size 分桶([8, 16, 24, 32, 40, …, 256]),prefill 固定 batch_size=1 并按 prefill_len 分桶([8, 16, 32, 64, 128, …, max_model_len]),桶大小后续按编译开销与端到端性能调优

## 优化点与收益
正文未给出量化数字。进度清单显示已完成:单芯片 TPU executor(#5292)、单主机张量并行(#5871)、多主机张量并行(#7457)、MoE 模型支持(#6457);未完成:INT8 量化、TPU 后端性能 benchmark 与优化。

## 现状与落地
已关闭。进度清单中主要目标(单芯片执行、单/多主机 TP、MoE)已勾选完成,INT8 量化与性能优化未勾选;整体落地情况需查关联 PR。

## 关联
- #5292
- #5871
- #7457
- #6457
- PyTorch XLA Llama 推理示例:https://github.com/pytorch-tpu/llama/tree/llama2-google-next-inference

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
