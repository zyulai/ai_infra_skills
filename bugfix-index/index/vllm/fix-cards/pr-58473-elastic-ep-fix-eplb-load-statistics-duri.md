# PR #58473: [Elastic EP] Fix EPLB load statistics during scaling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58473](https://github.com/vllm-project/vllm/pull/58473) |
| 作者 | itayalroy |
| 合并日期 | 2026-09-26 |
| labels | ready |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
Elastic EP 触发 reshuffle 时旧负载数据张冠李戴,导致 EPLB 把旧负载记到错误 expert;改用上次保存的每专家负载重排。

## 问题现象
EPLB 按"物理位置"记录专家负载;一次 reshuffle 之后,同一物理位置上可能已是另一个 expert,但旧的测量值仍留在历史窗口中。在 Elastic EP 场景下,新一轮测量窗口尚未积累完整(full window)时,弹性伸缩就可能再次强制触发 reshuffle,此时 EPLB 会把旧位置的负载数据统计到错误的 expert 上,做出错误的重排决策。正文未附具体错误日志。

## 根因
正文披露:EPLB 负载按物理位置记录,reshuffle 后物理位置与 expert 的对应关系已改变,而旧测量值未被清除;Elastic EP 在完整新测量窗口形成之前即可再次触发 reshuffle,导致重排依据的是过期甚至错误的负载历史。

## 修复方案
- `vllm/distributed/eplb/eplb_state.py`:
  - `EplbModelState` 新增 `last_expert_load` 字段(全局逻辑专家负载,add_model 时初始化为全 1,即等需求);
  - `rearrange()` 新增 `use_last_expert_load` 参数:开启时跳过对当前负载窗口的物理→逻辑映射,改用 rank 0 保存的 `last_expert_load` 经既有 all-reduce 分发;
  - 非 profile 路径在每次重排后把本次 `global_expert_load_window` 保存为 `last_expert_load`,供下次弹性 reshuffle 使用。
- `vllm/distributed/elastic_ep/elastic_execute.py`:`_perform_eplb_reshuffle` 中把按 `rank_mapping` 是否为 None 的两分支统一为:
```diff
-        if rank_mapping is None:
-            eplb_state.rearrange()
-        else:
-            eplb_state.rearrange(rank_mapping=rank_mapping)
+        eplb_state.rearrange(rank_mapping=rank_mapping, use_last_expert_load=True)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Elastic EP 触发的 reshuffle 路径全部改用上次保存负载,首次重排时 `last_expert_load` 为全 1(等需求假设),首窗口决策可能不同于旧实现;常规(非 Elastic)EPLB reshuffle 默认 `use_last_expert_load=False`,行为不变。正文明确指出常规 EPLB 在 `step_interval` 小于 `window_size` 时也会遇到同样的 stale-history 问题,但应另行修复,本 PR 不覆盖。

## 关联
正文未列出(cc @tlrmchlsmth @ilmarkov;diff 注释中引用了 #22167 的 review 讨论)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
