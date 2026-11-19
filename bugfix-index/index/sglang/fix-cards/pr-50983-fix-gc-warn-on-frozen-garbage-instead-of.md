# PR #50983: [Fix] gc: Warn on frozen garbage instead of reporting a leak

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#50983](https://github.com/sgl-project/sglang/pull/50983) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-11-17 |
| labels | documentation, run-ci, bypass-fail-fast, parallel-stages |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

调度器 GC freezer 此前把冻结的循环垃圾按"泄漏"上报(在 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 下会导致严格检查失败),但这部分垃圾很大程度来自改不了的第三方代码(如 `importlib.metadata` 缓存),本 PR 将其统一降级为 warning 日志,并删除只剩节流作用的 `SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK` 环境变量。

## 问题现象

正文披露:调度器的 GC freezer 会发现冻结堆中的循环垃圾;当走 `FailureReporter` 上报路径时,它们被当作泄漏处理,在 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 开启的 CI 严格检查中直接判为失败。实际案例:一次开启严格检查的 CI 运行失败,原因是 `importlib.metadata` 缓存——其 `FreezableDefaultDict` 持有一个指回自身的 lambda,形成循环垃圾。这类对象属于"我们无法修改的代码"产生的冻结垃圾,不应作为泄漏阻塞检查。

## 根因

正文披露:冻结堆里的循环垃圾并不等于真泄漏——冻结期间 gc 不会释放它们,refreeze 后收集即可;把这部分按 `Severity.LEAK` 上报使得无法避免的第三方循环引用(如 `importlib.metadata` 缓存的自引用 lambda)也触发严格失败,检查语义过严。另外 `SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK` 开关的剩余作用只剩"每次 idle freeze 都检查垃圾"而非"至多每 10 分钟检查一次",不再承担严格/宽松语义的开关职责。

## 修复方案

1. `GCFreezer`(`python/sglang/srt/managers/scheduler_components/gc_freezer.py`)改为只记录日志:
   - 构造参数移除 `enable_frozen_garbage_check`,不再创建 `FailureReporter`;
   - `_freeze` 中删除按开关的分支,统一走 `refreeze_gc` + 告警,间隔保持至多每 10 分钟(`GC_CHECK_INTERVAL_SECS`);
   - `_report_frozen_garbage` 更名为 `_warn_frozen_garbage`,统一 `logger.warning`,不再经 `FailureReporter` 上报。
2. `python/sglang/srt/managers/scheduler.py` 的 `init_gc_freezer` 同步去掉该参数。
3. 删除 `SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK`(`python/sglang/srt/environ.py`)及环境变量文档条目;`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 的注释更新为只覆盖 idle pool 检查与 KV page 检查发现的泄漏。
4. 单元测试同步改写:`test_frozen_garbage_warns_in_production_and_is_a_leak_under_the_env` 替换为 `test_frozen_garbage_is_checked_and_warned_at_most_once_per_interval`,不再断言 `Severity.LEAK` 上报。

关键 diff 摘录:

```diff
-    def _report_frozen_garbage(self, garbage: collections.Counter) -> None:
+    def _warn_frozen_garbage(self, garbage: collections.Counter) -> None:
         if not garbage:
             return
-        message = (
+        logger.warning(
             f"{sum(garbage.values())} frozen objects are cyclic garbage, "
             f"which a frozen heap never frees; collected them now. "
             f"Most common: {garbage.most_common(5)}"
         )
-        if self._enable_frozen_garbage_check:
-            self._reporter.report(Severity.LEAK, message)
-        else:
-            logger.warning(message)
```

```diff
-    SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK = EnvBool(False)
```

## 影响与回归风险

- 行为变化:冻结垃圾从"严格模式下可判失败的泄漏上报"变为固定 warning 日志;开启 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 的环境(含 CI)不再因第三方循环垃圾失败。
- 环境变量移除:仍设置 `SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK` 的部署/脚本不再有任何效果(该变量本就默认 `false`,正文说明其仅剩的作用是节流差异),环境变量文档已同步删除。
- 回归面:仅调度器 idle 冻结路径;冻结期间垃圾的释放时机(refreeze 收集)与堆增长检查 `_check_heap_growth` 逻辑不变。测试文件 `test/registered/unit/managers/scheduler_components/test_gc_freezer.py` 同步更新。

## 关联

- 相关开关:`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE`(不再作用于冻结垃圾)、`--gc-freeze-on-idle-secs`(GC freezer 的触发间隔,文档条目保留)。
- 相关实现:`python/sglang/srt/utils/gc_utils.py` 的 `freeze_gc` / `refreeze_gc`;`scheduler_components/state_check.py` 的 `FailureReporter` / `Severity`(不再被 gc_freezer 引用)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
