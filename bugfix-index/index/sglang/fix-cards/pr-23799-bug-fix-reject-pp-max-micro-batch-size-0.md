# PR #23799: [Bug Fix] Reject pp_max_micro_batch_size=0 to prevent silent deadlock on generate()

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23799](https://github.com/sgl-project/sglang/pull/23799) |
| 作者 | ppraneth |
| 合并日期 | 2026-04-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

显式传 `pp_max_micro_batch_size=0` 绕过 None 判断,调度永远分不出请求致 `engine.generate()` 无声死锁;启动断言拒绝 + 调度侧兜底。

## 问题现象

显式传入 `pp_max_micro_batch_size=0` 启动后,第一次 `engine.generate()` 调用无限挂起——没有断言、没有异常、没有 watchdog 触发,CUDA-graph capture 阶段之后 stderr 也不再有任何输出。原因是每个调度 tick 里 `Scheduler.get_num_allocatable_reqs` 返回 `0 - running_bs ≤ 0`,永远没有请求能从 waiting 转入 running。默认(None)不受影响,仅显式传 0 触发。

## 根因

正文披露:`pp_max_micro_batch_size` 是 `Optional[int]` CLI 参数,默认 None;为 None 时 `Scheduler.__init__` 会用 `max(self.max_running_requests // self.pp_size, 1)` 自动计算安全值(恒 ≥ 1)。显式传 0 绕过了 `scheduler.py` 里的 `is None` 判断,0 一路带进执行期,导致可分配请求数恒 ≤ 0、调度停摆。正文说明它与 `prefill_max_requests=0` 的无声死锁属同一家族的问题。

## 修复方案

双层防护:

1. `python/sglang/srt/server_args.py` `check_server_args`:新增断言,`pp_max_micro_batch_size` 必须为 None 或 ≥ 1,否则报 `AssertionError: pp_max_micro_batch_size must be a positive integer or None (for auto-compute). Got: 0`,在模型加载与 CUDA graph capture 之前快速失败;
2. `python/sglang/srt/managers/scheduler.py` `init_model_worker`:自动计算兜底条件从 `is None` 放宽为 `not pp_max_micro_batch_size`,0 也走安全自动计算路径——作为纵深防御,覆盖程序化调用 `Engine()` 绕过 CLI 校验的场景。

关键 diff 摘录(scheduler.py):

```diff
-        if get_global_server_args().pp_max_micro_batch_size is None:
+        if not get_global_server_args().pp_max_micro_batch_size:
             get_global_server_args().pp_max_micro_batch_size = max(
                 self.max_running_requests // self.pp_size, 1
             )
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅在启动参数校验与调度器初始化兜底两处收紧行为;合法取值(None 或 ≥ 1)行为不变,`pp_size > 1` 的既有断言链不受影响。

## 关联

- [#23789](https://github.com/sgl-project/sglang/issues/23789)(正文注明 Closes #23789)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
