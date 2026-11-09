# PR #15941: Fix incorrect running batch size in prefill stats

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15941](https://github.com/sgl-project/sglang/pull/15941) |
| 作者 | fzyzcjy |
| 合并日期 | 2026-01-03 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
prefill 统计日志里的 running batch size 不正确;`log_prefill_stats` 改为打点时实时取 `len(self.running_batch.reqs)`,不再使用过期的局部变量。

## 问题现象
正文未描述具体现象(Motivation/Modifications 均为空模板)。从标题可知:prefill 阶段输出的统计信息(statistics)中 running batch size 数值不正确。影响面是开启 scheduler 指标/日志时的观测数据,不影响调度与推理结果。

## 根因
正文未披露根因。(由 diff 推断)`get_new_batch_prefill` 末尾打印统计时,原代码把局部变量 `running_bs` 以位置参数形式传给 `log_prefill_stats(adder, can_run_list, running_bs, 0)`;该变量并非当前 `self.running_batch.reqs` 的实际长度(推测为早前计算、之后运行批次发生变化而过期),导致 prefill 统计中的 running batch size 与真实值不符。修复改为在打点处实时计算 `running_bs=len(self.running_batch.reqs)`。

## 修复方案
改 `python/sglang/srt/managers/scheduler.py` 的 `get_new_batch_prefill`:统计打印一行由位置参数改为关键字参数,`running_bs` 现场取 `len(self.running_batch.reqs)`,`running_bs_offline_batch=0` 显式命名。关键 diff 摘录:

```diff
         if self.current_scheduler_metrics_enabled:
-            self.log_prefill_stats(adder, can_run_list, running_bs, 0)
+            self.log_prefill_stats(
+                adder,
+                can_run_list,
+                running_bs=len(self.running_batch.reqs),
+                running_bs_offline_batch=0,
+            )
```

## 影响与回归风险
- 回归面:仅 `get_new_batch_prefill` 中统计日志/指标的取值来源,`log_prefill_stats` 接口签名不变;调度逻辑、请求处理与离线批次统计(仍传 0)均不变。风险极低,最坏情况是日志中 running batch size 数值仍与预期口径不一致(口径是否应包含某些请求正文未讨论)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
