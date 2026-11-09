# PR #50344: [BugFix] Scope divergent hybrid cache hits to capable connectors

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50344](https://github.com/vllm-project/vllm/pull/50344) |
| 作者 | ivanium |
| 合并日期 | 2026-08-09 |
| labels | bug, ready, v1, kv-connector |
| 热度 | 👍 3 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
per-group divergent hybrid cache hit 依赖 connector 补齐 Mamba state,并非所有 connector 都能做;引入 capability 门控,不能者回落 common local hit。

## 问题现象
正文描述:per-group hybrid cache hit 依赖 connector 恢复缺失的 Mamba(recurrent)state;把该策略套用到所有 connector 时,可能在 resume boundary 处报告 full-attention 命中却没有有效 state,即产生不一致的调度结果。触发场景是带 KV connector 的 hybrid(FA + Mamba)模型的 partial prefix cache hit 路径。

## 根因
正文披露:divergent per-group local cache hit 要求 connector 能在 local full-attention group 达到更深的 boundary 时补齐落后的 recurrent state,但该 lookup 策略此前对所有 connector 一视同仁,没有任何能力约定;不具能力的 connector 会报告一个在 resume boundary 无有效 state 的 full-attention 命中。

## 修复方案
- `KVConnectorBase_V1` 新增 `supports_divergent_local_hybrid_hits` property,默认 False(保守契约)。
- `NixlBaseConnector` 显式返回 True(opt in);未知/store 型 connector 保持默认 False。
- `MultiConnector` 仅当所有子 connector 都支持时才返回 True。
- `scheduler.py` 新增 `_get_local_prefix_cache_hit`:有 connector 且支持时才走 `kv_cache_manager.get_computed_blocks_for_connector`(divergent per-group lookup),否则走 `get_computed_blocks` 的 common local hit 并返回 `hit_diverged=False`;`schedule()` 中原来内联的 if/else 分支收敛为调用该函数。
- 测试:`test_partial_prefix_cache_hits.py` 19 passed;新增不可用/可用 connector 的单测,`test_hybrid_fa_deeper_hit_respects_connector_lookup_policy` 参数化覆盖 incapable connector 回落(16,6,32)与 capable 路径。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;作者说明这是对 #48425 所处理 divergence 的后续收口,覆盖 partial divergent hits。
- 回归面:所有带 KV connector 的 hybrid 模型调度路径;不支持的 connector(如未知/store 型、默认实现)从 divergent lookup 改回 common local hit,命中率策略变保守;NIXL 行为保持 divergent lookup;MultiConnector 取所有子 connector 的 AND。
- 正文注明:模型评估未跑,GPU hybrid-model 验证仍需补做。

## 关联
- #46455(正文说明非重复)
- #48195(正文说明非重复)
- #48425(原 divergence 处理)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
