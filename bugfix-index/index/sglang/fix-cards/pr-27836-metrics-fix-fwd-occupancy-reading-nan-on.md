# PR #27836: [Metrics] Fix `fwd_occupancy` reading NaN on every decode log line; probe-free `base-a`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27836](https://github.com/sgl-project/sglang/pull/27836) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-10 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
`update_device_timer` 在每个窗口起点把 `fwd_occupancy` 覆写为 NaN,与 decode 日志节奏锁相后每条 decode 日志都打印 nan;窗口起点改为保留上次发布值。

## 问题现象
开启 device timer(`SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`)时,decode 日志行的 `fwd occupancy` 可能持续为 nan:窗口计数器随所有 batch 结果推进,而 decode 日志节奏只数 decode batch,每个 prefill 都会让两者相位错开,纯 decode 阶段相位冻结,workload 可能恰好让每条 decode 日志都落在 NaN 刻度上,无限输出 `fwd occupancy: nan`。并发的 eval 负载(如 ngram spec CI 测试)因 prefill 不断推进相位而频繁触发。正文验证环境:Qwen3-8B + `--speculative-algorithm NGRAM`、`--decode-log-interval 10`、H200、40 个滚动请求的持续并发负载;main 分支 decode 行 nan 11/268 次打印(成串出现),本 PR 后仅 2/279 且孤立出现于 idle reset 后第一条 decode 打印(符合预期的 stale 语义);prefill 行 nan 30 次(idle stale 标记)不变。

## 根因
正文披露:`update_device_timer` 在每个 device-timer 窗口开始时(即 `_device_timer_window_batch_count == 0` 分支)直接把 `fwd_occupancy` gauge 置为 `float("nan")`;由于窗口边界与 decode 日志节奏可以相位锁定,一次性 NaN 会放大为每条日志都是 NaN。次要问题:occupancy 除法未防护 `cpu_time <= 0`,不推进的 timer 可能让调度循环崩溃(正文表述:"a non-advancing timer cannot crash the scheduler loop")。

## 修复方案
`python/sglang/srt/managers/scheduler_components/metrics_reporter.py` 的 `update_device_timer`:
```diff
         if self._device_timer_window_batch_count == 0:
             self._device_timer_window_start = now
             self._device_timer_window_gpu_time = 0.0
-            cpu_time = 0
-            self.fwd_occupancy = float("nan")
         else:
             cpu_time = now - self._device_timer_window_start
-            self.fwd_occupancy = min(
-                self._device_timer_window_gpu_time / cpu_time * 100, 100
-            )
+            if cpu_time > 0:
+                self.fwd_occupancy = min(
+                    self._device_timer_window_gpu_time / cpu_time * 100, 100
+                )
```
窗口起点保留上次发布值,仅 idle 后 `reset_device_timer_window` 导致真正 stale 时才发布 NaN;除法增加 `cpu_time > 0` 防护。配套 CI 改动:`_pr-test-stage.yml` 仅对 `base-a` stage 关闭 `SGLANG_ENABLE_ASYNC_ASSERT`(探针增加 GPU 工作、干扰 occupancy 测量),其余 stage 保持开启;`test_basic_sanity.py` 的 `fwd_occupancy_threshold` 97→99、`test_basic_sanity_eagle3.py` CUDA 侧 97→98(AMD 维持 80)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:按正文,所有计算出的 occupancy 值与之前 bit-identical,仅「无新样本的刻度发布什么」发生变化;`base-a` CI stage 不再注入 async-assert 探针(仅影响该 stage 的测量环境);sanity 阈值上调(97→99、97→98)使 occupancy 回归更早暴露,也可能提高 CI 对机器噪声的敏感度。

## 关联
- [Run #27310738345](https://github.com/sgl-project/sglang/actions/runs/27310738345)
- [Run #27310738262](https://github.com/sgl-project/sglang/actions/runs/27310738262)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
