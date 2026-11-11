# PR #44477: [Fix] metrics: Handle process CPU clock resets after restore

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#44477](https://github.com/sgl-project/sglang/pull/44477) |
| 作者 | rst0git |
| 合并日期 | 2026-11-09 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 31 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
CRIU 恢复后进程 CPU/monotonic 时钟回退,指标计数器出现负增量抛 ValueError 终止 scheduler;按时钟重置处理修复。

## 问题现象
CRIU restore 之后,进程 CPU 时钟可能低于 scheduler 保存的记账基线;由此产生的负增量会让 Prometheus 计数器抛 `ValueError`,在启用 metrics 时直接终止恢复出来的 scheduler。另外,若没有 time namespace,在 uptime 更短的宿主机上恢复还会让 monotonic 时钟倒退:scheduler 时间记账会停滞直到时钟追上,而 stage metrics 会累积负的 wall time,在 drain 时以同样的方式崩溃。

## 根因
正文披露:CRIU restore 会把进程 CPU 时钟重置到接近零,低于 scheduler 恢复前保存的记账基线,`elapsed_process_cpu_ns = now - start` 算出负值,负增量传入 Prometheus 计数器即抛 `ValueError`;同理,无 time namespace 时 monotonic 时钟在恢复后回退,`wall_ns - last_wall_ns` 为负,stage 记账累积负 wall time。正文未指认具体的引入提交。

## 修复方案
改 `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` 与 `python/sglang/srt/observability/scheduler_stage_metrics.py`:
- 把恢复后的 CPU 时钟读数当作「restore 以来的 CPU 时间」而非负增量,计数器永不递减;
- 当 monotonic 时钟低于上次采样(`clock_reset = now_wall_ns < accounting.last_sample_ns`)时,先 flush 缓冲的 idle 与 stage 总量,再对 scheduler 时间记账 rebase;保留当前 active stage,并重置 forward-occupancy 窗口(perf_counter 读同一时钟,device timer 窗口一并重启);
- `SchedulerStageMetricsRecorder._sample` 丢弃回退的时间段,drain 出的总量保持非负。

关键摘录(metrics_reporter.py):

```diff
         now_process_cpu_ns = time.process_time_ns()
         elapsed_process_cpu_ns = now_process_cpu_ns - accounting.start_process_cpu_ns
+        if elapsed_process_cpu_ns < 0:
+            # CRIU restore restarts the process CPU clock near zero, so its
+            # reading is the CPU time since restore. Never decrement the counter.
+            elapsed_process_cpu_ns = now_process_cpu_ns
```

并新增回归测试: idle/active 调度下的重复 CPU 时钟重置、monotonic 时钟重置、以及 reporting tick 之间重置时缓冲总量与 active stage 的保留。验证:6 个记账测试在隔离 source harness 通过,12 个重复重置用例用真实 Prometheus 计数器通过(idle/active 转换分别独立启用 metrics 与 device timing);本地因缺 `sgl_kernel` 无法直接跑测试。模型输出不受影响;速度测试不适用(仅影响 scheduler metrics 记账)。

## 影响与回归风险
- 影响窗口:正文未给出明确版本/提交范围;触发场景为 CRIU restore 后启用 metrics 的 scheduler。
- 回归面:仅 scheduler metrics 记账路径——`_record_scheduler_time` 的时钟重置分支(rebase、flush、occupancy/device timer 窗口重置)、process CPU 增量计算、`SchedulerStageMetricsRecorder._sample` 的回退 span 丢弃;正文明确推理执行不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
