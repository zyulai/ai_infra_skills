# PR #41423: [Bugfix] Fix spawn_new_process_for_each_test silently swallowing test failures

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41423](https://github.com/vllm-project/vllm/pull/41423) |
| 作者 | dzhengAP |
| 合并日期 | 2026-05-06 |
| labels | bug, nvidia, ready, rocm, v1 |
| 热度 | 👍 1 · 💬 17 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
spawn 装饰器在子进程跑 `python -m <module>` 而非测试函数,任何失败都静默通过;改为 cloudpickle 序列化函数经 stdin 执行,失败回传 traceback 抛 RuntimeError。

## 问题现象
`spawn_new_process_for_each_test` 形同虚设——无论测试函数做什么都通过,造成测试覆盖静默失效。issue 复现:
```python
@create_new_process_for_each_test("spawn")
def test_failing():
    raise ValueError  # 修复前永远通过
```

## 根因
正文披露:旧实现在子进程运行 `python -m <module_name>`,这会**重新执行模块的 `__main__` 块**而不是调用测试函数;`check_returncode()` 永远看到 exit 0,测试内抛出的任何异常都被静默吞掉。

## 修复方案
`tests/utils.py` 的 `spawn_new_process_for_each_test` 重写:用 `cloudpickle` 序列化测试函数与 args/kwargs(闭包可用),经 stdin 传给一个最小子脚本 `python -c`;子脚本调用 `f(*args, **kwargs)`,失败时把完整 traceback 写入临时文件并 exit 1,`Skipped` 以 exit 0 放行;父进程读 traceback(为空则回退到 stderr)并抛 `RuntimeError: Test subprocess '<name>' failed (exit code N):...`,使 CI 失败可行动。临时文件在 `finally` 中清理。这与已正确传播异常的 `fork_new_process_for_each_test` 对齐。关键 diff 摘录:
```python
-            cmd = [sys.executable, "-m", f"{module_name}"]
+            result = subprocess.run(
+                [sys.executable, "-c", child_script],
+                input=payload, capture_output=True, env=env,
+            )
+            if result.returncode != 0:
+                ...
+                raise RuntimeError(
+                    f"Test subprocess '{f.__name__}' failed "
+                    f"(exit code {result.returncode}):\n{tb}"
+                )
```
配套:新增 `tests/utils_/test_spawn_decorator.py`(通过/失败必抛 RuntimeError 的 xfail strict/pytest.skip 传播/参数化参数转发四个用例);装饰器现在真正生效后,`test_cudagraph_dispatch.py` 的类方法改为模块级函数、`test_custom_online.py` 的 async 测试改为同步函数内 `asyncio.run`(旧写法依赖装饰器吞行为)。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响所有使用 `spawn_new_process_for_each_test`/`create_new_process_for_each_test("spawn")` 的测试。
- 回归面:纯测试基础设施(`tests/utils.py`)与三个测试文件的重构,不触及产品代码。直接后果:此前被静默吞掉的 spawn 测试失败现在会真实报红——某些一直"绿着"的用例(cudagraph dispatch、custom logitsprocs)可能暴露潜在失败。子进程内 `pytest.skip` 正确传播为通过;失败路径包含完整子进程 traceback。

## 关联
- https://github.com/vllm-project/vllm/issues/41415(所修 issue,含复现)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
