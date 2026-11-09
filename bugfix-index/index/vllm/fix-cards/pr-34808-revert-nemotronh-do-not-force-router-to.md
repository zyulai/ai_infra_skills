# PR #34808: Revert "[NemotronH] Do not force router to run in fp32 (#34582)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34808](https://github.com/vllm-project/vllm/pull/34808) |
| 作者 | roikoren755 |
| 合并日期 | 2026-02-19 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论
回滚 #34582:路由器不再强制 fp32 导致 NemotronH 精度退化;恢复 fp32 router,更优实现另在 #34302 跟进。

## 问题现象
NemotronH 模型出现精度退化(accuracy degradation),由 #34582 引入。正文未提供具体的评测数值或对比表,Test Plan 与 Test Result 均为空。

## 根因
正文披露:#34582("[NemotronH] Do not force router to run in fp32")为性能优化而取消了路由器的 fp32 强制,随之引入精度退化。(由 diff 推断)回滚把 gate(`ReplicatedLinear`)的 `params_dtype` 显式设为 `torch.float32`、进入 gate 前把 `hidden_states` 转为 fp32,并把 `router_logits_dtype` 传给 `SharedFusedMoE`,即恢复 fp32 路由精度;这印证了 #34582 的性能收益是以路由计算精度为代价的。

## 修复方案
直接 revert #34582 对 `vllm/model_executor/models/nemotron_h.py` 的改动,恢复 fp32 路由:

```diff
+        router_logits_dtype = torch.float32
         self.gate = ReplicatedLinear(
             config.hidden_size,
             config.n_routed_experts,
             bias=False,
+            params_dtype=router_logits_dtype,
             quant_config=None,
             prefix=f"{prefix}.gate",
         )
...
-        router_logits, _ = self.gate(hidden_states)
+        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
```

同时把 `router_logits_dtype=router_logits_dtype` 传入 `SharedFusedMoE` 构造。

## 影响与回归风险
- 影响窗口:#34582 合并之后至本回滚之前,NemotronH 的 MoE 路由精度退化。
- 回归面:`nemotron_h.py` 的 MoE 路由路径——gate 权重精度、路由输入精度与传给 `SharedFusedMoE` 的 `router_logits_dtype` 全部恢复 fp32,精度回到 #34582 之前;相应地,#34582 追求的性能收益被放弃。正文说明正与 @robertgshaw2-redhat 在 #34302 中设计更好的实现方式,届时可再次尝试该优化。

## 关联
- PR #34582(被回滚的改动)
- PR #34302(跟进的更优实现)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
