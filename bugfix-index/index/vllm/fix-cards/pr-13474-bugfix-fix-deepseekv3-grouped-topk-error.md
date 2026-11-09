# PR #13474: [Bugfix] Fix deepseekv3 grouped topk error

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#13474](https://github.com/vllm-project/vllm/pull/13474) |
| 作者 | chenxb002 |
| 合并日期 | 2025-02-20 |
| labels | ready |
| 热度 | 👍 6 · 💬 6 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
fused MoE 的 grouped topk 与 DeepSeek V3 官方逻辑不一致:bias 场景分组打分应为组内 top-2 求和、masked 分数应置 -inf 而非 0。

## 问题现象
正文未描述具体运行时报错现象;说明的是计算逻辑偏差:vllm 的 grouped_topk 与官方代码存在差异,带 bias 时的分组方式不对,且 masked 分数置 0 可能在未 mask 组分数为负时选中被 mask 组的专家,导致某些场景下专家选择错误或次优。影响 DeepSeek V3 类模型的 fused MoE 路由。

## 根因
正文披露两点与官方推理代码的差异:
1. 当新引入的 bias 项(`e_score_correction_bias`)非 None 时,应先取每组 top-2 分数求和、再据此选 top-k 组;vllm 却用组内 max;
2. 被 mask 的分数被置 0,官方 DeepSeek V3 代码将 masked 分数重置为 -inf。

## 修复方案
`vllm/model_executor/layers/fused_moe/fused_moe.py` 的 `grouped_topk`:

```diff
-    group_scores = scores.view(num_token, num_expert_group,
-                               -1).max(dim=-1).values  # [n, n_group]
+        group_scores = (scores.view(num_token, num_expert_group,
+                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
+    else:
+        group_scores = scores.view(num_token, num_expert_group,
+                                   -1).max(dim=-1).values  # [n, n_group]
```

以及 masked 填充值改为 `float("-inf")`(原为 0.0);`num_token = scores.shape[0]` 移到 bias 分支之前以便两个分支共用。bias 为 None 时保留原 max 逻辑。

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:`grouped_topk` 的所有调用方;带 bias 的 MoE(如 DeepSeek V3)分组打分与专家选择结果改变,mask 填充从 0 改为 -inf;无 bias 的模型分组逻辑不变。

## 关联
[DeepSeek V3 官方推理代码 model.py L583-L586](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L583-L586)、[L589(masked 分数置 -inf)](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L589)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
