# RFC #24112: [RFC]:  Improve MoE triton kernel tuning

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24112](https://github.com/vllm-project/vllm/issues/24112) |
| 作者 | jeejeelee |
| 创建日期 | 2025-09-02 |
| 状态 | ⚪ closed(关闭于 2026-01-26) |
| labels | RFC, stale |
| 热度 | 👍 6 · 💬 11 |
| 分类 | MoE |

## 一句话结论
提议按 triton 版本分目录管理 MoE tuned config,并补充 `benchmark_moe` 调优文档,解决不同 triton 版本调优配置不一致影响性能的问题。

## 提案动机
不同 triton 版本会产出不同的 MoE tuned configuration。例如 triton `3.4.0` 的 tuned config 与 vLLM 0.10.1.1 版本仓库中对同一配置(E=128,N=768,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128])的 config 文件并不相同,这些差异可能影响模型性能。RFC 主要讨论如何管理不同 triton 版本的 tuned config。

## 提出的方案
- 把 MoE tuned config 拆分到 N 个文件夹:目前已有 `legacy_configs` 与 `triton_3_4_0` 两个目录——`legacy_configs` 存放无法追溯到具体 triton 版本的配置,`triton_3_4_0` 对应当前 triton 版本 3.4.0;随 triton 版本更新,未来会出现更多 triton 版本目录(见 PR #24113)。
- 补充与 `benchmark_moe` 相关的文档,包括如何做 benchmark、如何通过 `VLLM_TUNED_CONFIG_FOLDER` 传入本地 tuned configs 等(见 PR #24860),以鼓励更多用户在自己的硬件上调优 kernel。

## 优化点与收益
正文未给出量化数字。收益是正确性与可维护性层面的:避免 triton 版本升级后继续使用不匹配的 tuned config 导致的 MoE kernel 性能损失,并通过文档降低用户为自有硬件产出 tuned config 的门槛。

## 现状与落地
已关闭(2026-01-26)。正文中两项提议均给出了对应 PR(#24113 目录拆分、#24860 调优文档),说明方案已有配套实现;最终落地情况需查关联 PR。

## 关联
- PR #24113: https://github.com/vllm-project/vllm/pull/24113
- PR #24860: https://github.com/vllm-project/vllm/pull/24860
- triton 3.4.0 tuned config 示例: https://github.com/jeejeelee/vllm/blob/improve-moe-tune-config/vllm/model_executor/layers/fused_moe/configs/triton_3_4_0/E%3D128%2CN%3D768%2Cdevice_name%3DNVIDIA_H20-3e%2Cdtype%3Dfp8_w8a8%2Cblock_shape%3D%5B128%2C128%5D.json

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
