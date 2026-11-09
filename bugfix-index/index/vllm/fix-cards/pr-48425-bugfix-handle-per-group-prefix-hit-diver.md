# PR #48425: [BugFix] Handle per-group prefix-hit divergence for hybrid models with KV connector

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48425](https://github.com/vllm-project/vllm/pull/48425) |
| 作者 | njhill |
| 合并日期 | 2026-07-23 |
| labels | bug, ready, v1, kv-connector |
| 热度 | 👍 4 · 💬 6 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
混合模型 + KV connector 下各 cache 组 prefix 命中深度可能分叉,调度器按 FA 命中续算会踩到已驱逐块;无外部 token 时回退到一致边界。

## 问题现象
hybrid(Mamba + full-attention)模型搭配 KV connector 时,在块压力下调度可能出现错误恢复:按 per-group 查询得到的某组更深命中被当作本地已计算继续调度,而该边界对应的 FA 块或 Mamba 状态实际已不在缓存中,新测试文档描述其后果为 engine crash / dirty KV。正文本身只写了「This is a fix for #46453」,未展开现象细节;触发场景(FA 尾部被驱逐而更深的 Mamba 状态存活,或相反)来自补丁中新增回归测试的描述。

## 根因
正文未披露根因,仅指向 issue #46453。(由 diff 推断)机制如下:hybrid 模型下 FA 组与 Mamba 组的 prefix 命中深度可能分叉——块压力下 FA 前缀尾部可能被驱逐,而更深的 Mamba 状态块存活(或相反)。此前 scheduler 在 connector 路径取 `max(per_group_hits)` 作为本地已计算 token 数,当某组命中深于 FA 组时,报告的更深命中在本地并没有有效状态;只有 connector 恰好供给外部 token 时才成立,否则续算边界处 Mamba 状态无效,导致崩溃或脏输出。反向情形(FA 更深而 connector 无外部命中)则会在无有效 Mamba 状态的边界静默恢复,产生错误输出。

## 修复方案
三处改动(按 diff):
1. `vllm/v1/core/kv_cache_coordinator.py`:在分组校验后记录 `full_attention_group_id`(无 full-attention 层时为 None),作为 per-group 查询的参照组。
2. `vllm/v1/core/kv_cache_manager.py`:新增 `get_computed_blocks_for_connector`,执行 `find_longest_cache_hit_per_group`,任一组命中深于 FA 组即回退 `get_computed_blocks` 用一致边界;否则报 FA 命中并附带 `hit_diverged` 标志(`min(per_group_hits) < num_local`)。
3. `vllm/v1/core/sched/scheduler.py`:connector 路径改走新方法;当 `hit_diverged` 且外部 token 数为 0 时,回退 `get_computed_blocks` 收敛到各组一致的边界。

同时新增两个调度器回归测试:FA 尾部被驱逐但 Mamba 深状态存活(有/无外部命中两种参数),断言调度回退到收敛边界。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;按 diff,受影响的是 scheduler 中既有的 connector + hybrid per-group 查找路径(即被本次替换的 `max(per_group_hits)` 逻辑)。
- 回归面:改动集中在 KV connector 调度路径——`schedule()` 中 `num_computed_tokens == 0` 的 prefix 查找分支;非 hybrid 模型与已收敛命中仍走原 `get_computed_blocks`。行为变化:分叉且无外部命中时改为重算更多 token(正确性优先,吞吐略降);connector 供数路径(如 nixl 无条件传输 Mamba 状态)的假设在方法文档中已写明,依赖 FA 命中作为本地长度仍保留。

## 关联
- [#46453](https://github.com/vllm-project/vllm/issues/46453)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
