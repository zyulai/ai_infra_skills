# RFC #26108: [RFC]: MoE shared_experts fusion with the selected experts inside FusedMoE

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26108](https://github.com/vllm-project/vllm/issues/26108) |
| 作者 | alexm-redhat |
| 创建日期 | 2025-10-02 |
| 状态 | ⚪ closed(关闭于 2026-01-31) |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 2 |
| 分类 | MoE |

## 一句话结论
提议把 DeepSeek 类 MoE 的 shared_experts 与 selected experts 合并进单个 FusedMoE 执行,削减 decode 阶段 shared_experts 的独立开销。

## 提案动机
DeepSeekR1 的 profile 结果显示,decode 迭代中有相当一部分时间花在 selected_experts 的 FusedMoE 之前执行的 shared_experts 上。为降低该开销,可把 shared_experts 与 selected_experts 融合(合并权重矩阵)后作为单个 FusedMoE 部分执行。正文附 DeepSeekR1-FP4 与 FP8 在 8xB200、batch-size 32 下的单次 transformer decode 迭代操作分解(flags:VLLM_USE_FLASHINFER_MOE_FP8=1、VLLM_FLASHINFER_MOE_BACKEND="latency")。

## 提出的方案
- 合并 shared_experts 与 selected_experts 的权重矩阵,统一走单个高度优化的 FusedMoE 部分,不再用 deepseek_v2.py 的 DeepseekV2MLP 类
- 收益点:1) 去掉当前 shared experts 中的 FP16=>FP4/FP8 数据转换(FusedMoE 部分本来就要做);2) 完全消除 shared experts 中由 2 个 elementwise op 组成的 silu;3) shared_experts 进入高度优化的 FusedMoE 执行
- DP 模式下 shared_experts 与通信重叠,应保持与 selected_experts 分离;融合收益主要适用于 TP 场景
- 另指出 FusedMoE kernel 的 pre/post ops(此执行中来自 TRTLLM flashinfer 代码库)有开销,应设法降低

## 优化点与收益
正文引用此前尝试该融合的 PR #15502(因量化 op 问题未合入):其中提到的提升为 34%(作者推测真实提升约 15%)。RFC 本身未给出新的端到端量化数字,仅附 profile 分解图与上述定性收益。

## 现状与落地
已关闭(2026-01-31),labels 含 stale;正文未说明是否实现,落地情况需查关联 PR(PR #15502 当年未合入)。

## 关联
- PR #15502: https://github.com/vllm-project/vllm/pull/15502

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
