# PR #15579: Revert "[FusedMoE] Fix fused w13 tp sharded weight loading"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15579](https://github.com/sgl-project/sglang/pull/15579) |
| 作者 | hnyls2002 |
| 合并日期 | 2025-12-21 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Parallelism / Communication |
| bug 类型 | regression |

## 一句话结论
#15432 对 fused w13 TP 分片加载的特殊处理引发报错,整体回滚恢复原先统一的 narrow 分片逻辑。

## 问题现象
正文未描述具体用户侧现象,仅附上一个 CI 失败链接(#15509 相关 run 的 step 5 报错),说明 #15432 合入后引发错误(Errors),遂执行回滚。

## 根因
正文未披露根因分析,仅声明 "Reverts sgl-project/sglang#15432" 并附错误发生的 CI 日志链接。
(由 diff 推断)被回滚的 #15432 在 `_load_w13` 中为 `shard_id == "w13"` 增加了特殊路径:把 fused 权重 [w1; w3] 按一半宽度分别对 w1、w3 独立 `narrow` 取各 TP rank 的分片,再用 `torch.cat` 拼回 [w1_shard; w3_shard];回滚后所有情况统一走原先的 `loaded_weight.narrow(shard_dim, shard_size * tp_rank, shard_size)`。即 #15432 的分片方式与实际加载/映射约定不符导致报错,回滚恢复旧分片行为。

## 修复方案
原样回滚 #15432,改动集中于 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的 `_load_w13`:删除 shard_id=="w13" 的独立分片+拼接分支(约 23 行),恢复统一的 narrow:
```diff
-                if shard_id == "w13":
-                    single_shard_size = shard_size // 2
-                    ...
-                    w1_shard = loaded_weight.narrow(
-                        shard_dim, single_shard_size * tp_rank, single_shard_size
-                    )
-                    w3_shard = loaded_weight.narrow(
-                        shard_dim, full_single_size + single_shard_size * tp_rank, ...
-                    )
-                    loaded_weight = torch.cat([w1_shard, w3_shard], dim=shard_dim)
-                else:
-                    loaded_weight = loaded_weight.narrow(
-                        shard_dim, shard_size * tp_rank, shard_size
-                    )
+                loaded_weight = loaded_weight.narrow(
+                    shard_dim, shard_size * tp_rank, shard_size
+                )
```

## 影响与回归风险
- 影响窗口:正文未说明;按 revert 对象,#15432 合并之后、本 PR 之前的提交受影响。
- 回归面:fused_moe_triton 的 MoE 权重 TP 分片加载路径(`_load_w13`,影响所有经该路径加载 w1/w3 的 MoE 模型);#15432 原本想解决的 fused w13 分片加载问题(如有)会随之回归到旧状态;后续正确修法需另行重做(正文无后续计划说明)。

## 关联
- https://github.com/sgl-project/sglang/pull/15432 (被回滚的 PR)
- 失败 CI:https://github.com/sgl-project/sglang/actions/runs/20405657770/job/58655183248?pr=15509 (正文链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
