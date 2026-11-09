# PR #18992: [BugFix] Fix incorrect metrics shutdown error log message

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18992](https://github.com/vllm-project/vllm/pull/18992) |
| 作者 | njhill |
| 合并日期 | 2025-06-01 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

非多进程模式下 shutdown 误调 mark_process_dead 报错,现按路径判空提前返回。

## 问题现象

在未启用 prometheus 多进程模式(即 API server 数量为 1)时,进程关闭阶段 metrics 清理会打印错误日志:

```
ERROR 05-31 01:04:14 [prometheus.py:77] Error during metrics cleanup: expected str, bytes or os.PathLike object, not NoneType
```

触发条件是单 API server 的常规部署在 shutdown 时,影响面为日志噪音/清理逻辑失败,不影响服务运行期。

## 根因

按正文披露:该问题由 PR #17546 引入——shutdown 时无条件调用 `multiprocess.mark_process_dead(pid)`,而 `mark_process_dead` 只应在使用 prometheus 多进程模式(多于 1 个 API server)时调用。单进程模式下多进程目录 `_prometheus_multiproc_dir` 为 None,传入后抛出 "expected str, bytes or os.PathLike object, not NoneType",被 except 捕获后打出上述错误日志。

## 修复方案

修改 `vllm/v1/metrics/prometheus.py` 的 `shutdown_prometheus()`:先取模块级变量 `_prometheus_multiproc_dir`,为 None(非多进程模式)时直接 return,不再执行清理;非 None 时把该路径显式传给 `mark_process_dead(pid, path)`。

关键 diff 摘录:

```diff
 def shutdown_prometheus():
     """Shutdown prometheus metrics."""
+
+    path = _prometheus_multiproc_dir
+    if path is None:
+        return
     try:
         pid = os.getpid()
-        multiprocess.mark_process_dead(pid)
+        multiprocess.mark_process_dead(pid, path)
         logger.debug("Marked Prometheus metrics for process %d as dead", pid)
     except Exception as e:
         logger.error("Error during metrics cleanup: %s", str(e))
```

## 影响与回归风险

- 影响窗口:自引入 PR #17546 的版本起至本修复前,单 API server 部署在 shutdown 时都会出现该错误日志。
- 回归面:仅 `shutdown_prometheus` 一处路径。多进程模式(配置了 `_prometheus_multiproc_dir`)的行为不变,且现在显式传路径更正确;单进程模式 shutdown 时不再执行任何 metrics 清理,也不再有异常日志。风险点是若未来存在目录变量未初始化但确需清理的场景,会被跳过。

## 关联

- [PR #17546](https://github.com/vllm-project/vllm/pull/17546)(问题引入来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
