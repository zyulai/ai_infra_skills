# PR #13694: Fix: CI monitor should not exit with error on regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13694](https://github.com/sgl-project/sglang/pull/13694) |
| 作者 | alisonshao |
| 合并日期 | 2025-11-21 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
CI monitor 夜间汇总检测到回归时不再以退出码 1 报错退出,改为打印提示后正常退出(0)。

## 问题现象
正文为空,未描述具体现象。从标题可知:CI monitor 在检测到回归(regressions)时会以错误码退出,这是需要修复的行为。由 diff 推断(由 diff 推断),触发场景是夜间分析模式(nightly summary)运行结束且 `detect_nightly_regressions` 检出回归时,进程 `sys.exit(1)` 非零退出。

## 根因
正文未披露根因。(由 diff 推断)机制:`scripts/ci_monitor/ci_analyzer.py` 的 `main()` 中,夜间汇总分支在检出回归时执行 `sys.exit(1)`,使 monitor 进程本身以失败状态结束——回归是监控要"报告"的对象,而不是监控程序自身的故障;修复者将其改为打印警告并统一 `sys.exit(0)`,即"报告回归但不停止监控"。

## 修复方案
改动文件 `scripts/ci_monitor/ci_analyzer.py` 的 `main()`:夜间汇总分支中,检出回归时不再 `sys.exit(1)`,改为打印 "\n⚠️  Regressions detected - see report above";无回归分支原 `sys.exit(0)` 移出,两个分支统一以 `sys.exit(0)` 结束。

关键 diff:
```diff
-            # Exit with error code if regressions detected
+            # Report regressions but don't stop the monitor
             if regressions:
-                sys.exit(1)
+                print("\n⚠️  Regressions detected - see report above")
             else:
                 print("\n✓ No significant regressions detected")
-                sys.exit(0)
+            sys.exit(0)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间。
- 回归面:仅 `ci_analyzer.py` 夜间汇总模式的进程退出码与输出文案;回归信息仍会打印在报告中,但不再反映在退出码上,依赖退出码判断夜间分析是否检出回归的上游调度/告警逻辑将无法再通过退出码感知回归。常规 CI 分析模式("Regular CI analysis mode")分支不在改动范围内。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
