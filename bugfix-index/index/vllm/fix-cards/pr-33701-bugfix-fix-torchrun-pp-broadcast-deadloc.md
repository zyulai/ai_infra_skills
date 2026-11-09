# PR #33701: [Bugfix] Fix torchrun PP broadcast deadlock with async scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33701](https://github.com/vllm-project/vllm/pull/33701) |
| 作者 | Isotr0py |
| 合并日期 | 2026-02-04 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
torchrun PP 已在 logits 计算时广播输出,async scheduling 下再广播 sampled token 造成死锁;跳过重复广播。

## 问题现象
torchrun + PP + async scheduling 下发生死锁。此前测试里以 `async_scheduling=False` 规避(diff 中原有 FIXME 注释:「async scheduling causes deadlock on torchrun with PP」),本 PR 移除该规避并修复根因(issue #33533 的根因)。正文未描述完整堆栈。

## 根因
正文披露:torchrun(external_launcher)PP 模式已经在 logits 计算阶段把 PP 输出广播到所有 rank(对应 gpu_model_runner.py 中 `broadcast_pp_output=True` 的路径),因此在 `sample_tokens` 里再对 sampled token ids 做一次 `_pp_broadcast_prev_sampled_token_ids` 广播是多余的;async scheduling 下这次多余的广播调用导致 rank 间相互等待、死锁。

## 修复方案
`vllm/v1/worker/gpu_model_runner.py` `sample_tokens`:在 `use_async_scheduling` 且 PP 末 rank 的分支上加 `not self.broadcast_pp_output` 条件,torchrun 外部启动器(broadcast_pp_output=True)不再重复广播:

```diff
-            if pp.world_size > 1 and pp.is_last_rank:
+            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                 self._pp_broadcast_prev_sampled_token_ids(
                     sampler_output.sampled_token_ids
                 )
```
测试:`tests/distributed/test_torchrun_example.py` 与 `test_torchrun_example_moe.py` 删除 `async_scheduling=False` 及 FIXME 注释,恢复默认异步调度。

## 影响与回归风险
- 影响窗口:正文未说明(async scheduling 支持 PP 期间,#33650 的后续)。
- 回归面:仅影响 async scheduling + PP 的末 rank 广播分支;`broadcast_pp_output=True`(torchrun 外部启动器)路径跳过第二次广播,普通 Ray/多进程 PP 路径仍广播;需确认非 async 场景行为不变(diff 只改 async 分支)。

## 关联
- Fixes 根因 issue [#33533](https://github.com/vllm-project/vllm/issues/33533)
- 前置 PR [#33650](https://github.com/vllm-project/vllm/pull/33650)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
