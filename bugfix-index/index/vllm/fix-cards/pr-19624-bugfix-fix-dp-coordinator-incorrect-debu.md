# PR #19624: [BugFix] Fix DP Coordinator incorrect debug log message

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19624](https://github.com/vllm-project/vllm/pull/19624) |
| 作者 | njhill |
| 合并日期 | 2025-06-14 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
DP coordinator 波次切换 debug 日志把目标波次少打 1,显示 from N to N;先算 new_wave 再打日志修正。

## 问题现象
v1 数据并行(DP)coordinator 在收到 wave_complete 通知、推进波次时,debug 日志输出为 `"Moving DP wave from N to N"`,而正确语义应为 `"Moving DP wave from N to (N + 1)"`。触发场景为引擎进入全局暂停状态、由 rank 0 引擎发出 wave_complete 消息后的波次推进路径。该问题只影响调试日志内容,正文未描述功能性影响。

## 根因
正文直接披露:日志内容打印错误,打印的是 `"Moving DP wave from N to N"` 而非 `"Moving DP wave from N to (N + 1)"`。具体机制由 diff 推断:原代码在 `wave_complete` 分支中,日志直接打印 `self.current_wave` 与旧波次 `wave` 两个变量,而实际赋值的是 `wave + 1`,即日志里的"目标波次"少了 1,故两次打印值相同。

## 修复方案
改动 `vllm/v1/engine/coordinator.py` 的 `process_input_socket`:

1. 主修复:先计算 `new_wave = wave + 1`,日志与赋值统一使用该变量。关键 diff 摘录:

```python
if self.current_wave <= wave:
    new_wave = wave + 1
    logger.debug("Moving DP wave from %d to %d.",
                 self.current_wave, new_wave)
    self.current_wave = new_wave
    self.engines_running = False
```

2. 顺带的相邻小改进:将「wave 过期时置 `engine_to_exclude = None`」的判断移入 `if not self.engines_running:` 分支内部;并提取 `scheduler_stats` 局部变量、修正一处注释标点。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:按 diff 实际改动,除日志文本修正外,`engine_to_exclude` 的清空条件从「wave 过期即清空」收窄为「仅当 `engines_running` 为 False 且 wave 过期时清空」,属行为变化,可能影响 DP 启动/暂停切换期间对陈旧波次消息的广播范围;`scheduler_stats` 提取为局部变量为纯重构。

## 关联
正文未列出 issue/PR 链接(感谢 @SageMoore 发现该问题)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
