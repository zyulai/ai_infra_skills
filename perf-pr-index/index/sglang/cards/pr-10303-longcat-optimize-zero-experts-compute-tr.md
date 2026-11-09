# PR #10303: [LongCat] Optimize zero_experts_compute_triton by changing mask

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zk-lover |
| 合并 | 2025-09-11 |
| 分类 | MoE(major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/10303 |

## 一句话结论

LongCat zero-expert 的掩码值从 0 改为 -1,消除与真实 expert 0 的混淆,prefill 吞吐提升 10%。

## 动机

LongCat flash 引入的 zero expert 不参与实际专家计算,其 token 的 expert ID 越界(`expert_indices >= num_experts`)。`zero_experts_compute_triton` 原先把这些越界 ID 掩成 0,会与 ID 本来就是 0 的真实 expert 的 token 混淆;作者发现调整 expert mask 的方式存在优化空间。

## 优化点

改动为 `python/sglang/srt/layers/moe/ep_moe/kernels.py` 中 `zero_experts_compute_triton` 的一行:越界的 normal_expert_mask 行的 `expert_indices` 从置 0 改为置 -1(`expert_scales` 已同时置 0)。这样 zero-expert token 不再被记到 expert 0 名下参与后续按专家的分组/计算流程,只保留其 scale 加权路径;属于 MoE 路由元数据层面的修正,不触及内核算术。

## 关键改动

```text
python/sglang/srt/layers/moe/ep_moe/kernels.py: zero_experts_compute_triton 中 expert_indices[normal_expert_mask] = 0 改为 = -1
```

## 性能声明

PR 正文仅声明"我们的模型 prefill 阶段吞吐提升 10%",未注明模型规模、硬件、workload 或基准命令等测试条件;Benchmarking and Profiling 章节为空模板。

## 验证与回归风险

PR 的 Accuracy Tests 章节为空模板,未提供精度验证结果。风险面:改动影响所有启用 zero-expert 路径的模型的 `zero_experts_compute_triton` 行为;-1 哨兵值依赖下游分组逻辑正确跳过负 ID(素材中未展示下游对 -1 的处理,无法进一步核实);对不含 zero expert 的模型无影响(掩码行不存在)。

## 关联

无(素材未列出前序/后续 PR 编号;co-review 邀请 @Orchard-DT)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
