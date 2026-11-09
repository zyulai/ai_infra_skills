# PR #8752: fix: w4afp8 accuracy problem and rebase

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8752](https://github.com/sgl-project/sglang/pull/8752) |
| 作者 | yangsijia-celina |
| 合并日期 | 2025-08-11 |
| labels | (none) |
| 热度 | 👍 6 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
w4afp8 CUTLASS MoE 精度修复:新增专用 post-reorder triton kernel 按 local topk 正确聚合,补权重 scale 的 GROUP 注册与 routed_scaling_factor 缩放。

## 问题现象
正文未描述具体现象:PR 正文的 Motivation/Modifications/Accuracy Test 各节均为模板注释、未填写任何内容,只有标题「fix: w4afp8 accuracy problem and rebase」与 diff 可用,并注明 co-author 为 @ayrnb。

## 根因
正文未披露根因。从 diff 可描述改动所暗示的机制(由 diff 推断):`cutlass_w4a8_moe` 原先复用通用的 `post_reorder_triton_kernel`,传入 `topk_ids_`/`start_expert_id`/`end_expert_id` 做 expert 归属判断与 dst 偏移,而 `w4afp8.py` 的 `apply` 里另有按 `layer.expert_map` 把 topk_ids 翻译成本地 id 的分支——两处对「非本地 expert」的表示方式与 kernel 的聚合假设不一致,导致按 topk 权重聚合中间结果时取错 expert/偏移,产生精度问题;此外 `w13_weight_scale` 未注册 GROUP 类型的 quant_method,权重 scale 可能无法按 group 正确加载。

## 修复方案
三处改动:

1. `ep_moe/kernels.py` 新增 `post_reorder_triton_kernel_for_cutlass_moe`:按 `local_topk_ids` 判断(`expert_id != num_experts` 才聚合),`dst_idx = src2dst - dst_start` 后加权累加到输出;
2. `cutlass_w4a8_moe.py` 改用该新 kernel,参数从 `topk_ids_`/`start_expert_id`/`end_expert_id` 换成 `local_topk_ids`/`num_experts`;
3. `w4afp8.py`:`create_weights` 中为 `w13_weight_scale` 补注册 `FusedMoeWeightScaleSupported.GROUP` 的 quant_method;`apply` 签名扩展(`x`、`activation`、`apply_router_weight_on_input`、`routed_scaling_factor`),删掉 `layer.expert_map` 翻译分支,统一为:

```python
local_topk_ids = torch.where(
    topk_ids == -1,
    layer.num_experts,
    topk_ids,
)
```

并在返回前对输出乘 `routed_scaling_factor`(非 None 时)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:cutlass w4a8 MoE 的专家并行(EP)推理路径;w4afp8 量化方法的权重加载(scale 按 GROUP 注册改变 checkpoint scale 的加载映射)与 `apply` 调用方接口(新增参数,调用处需适配);expert_map 相关逻辑被移除,依赖 expert_map 语义的场景行为改变。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
