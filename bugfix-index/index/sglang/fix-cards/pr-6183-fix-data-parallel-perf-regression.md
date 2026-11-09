# PR #6183: Fix data parallel perf regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6183](https://github.com/sgl-project/sglang/pull/6183) |
| 作者 | merrymercy |
| 合并日期 | 2025-05-11 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Parallelism / Communication |
| bug 类型 | regression |

## 一句话结论
#4499 在 data_parallel_controller 引入忙等 while 循环空耗 CPU,改为长 sleep 修复性能回归。

## 问题现象
正文未描述具体现象(无具体性能数字或日志),仅说明 #4499 引入了 data parallel 性能回归。

## 根因
正文披露:PR #4499 在 `data_parallel_controller.py` 中加入了一个 busy while loop(忙等待循环),造成数据并行性能回归;本 PR 改用 sleep 而非忙等。

## 修复方案
修改 `python/sglang/srt/managers/data_parallel_controller.py` 的 `launch_tensor_parallel_group_thread`:把保活线程里的 `while True: pass` 忙等待改为 `time.sleep(30 * 24 * 3600)`,线程仍常驻但不再空转烧 CPU。正文注释说明该线程不能关闭,否则 scheduler.py 里的 `kill_itself_when_parent_died` 会杀掉 scheduler。另附带调整 `.github/workflows/execute-notebook.yml`:移除 actions/setup-python 步骤,改用 `ln -sf "$(which python3)" /usr/bin/python`。

```diff
         while True:
-            pass
+            time.sleep(30 * 24 * 3600)
```

## 影响与回归风险
- 影响窗口:#4499 合入后至本 PR 之前(正文明确该 PR 引入回归)。
- 回归面:数据并行 controller 的保活线程行为(改为长 sleep 后功能不变、CPU 占用消除);另影响 execute-notebook CI workflow 的 Python 环境设置方式。

## 关联
- sgl-project/sglang#4499(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
