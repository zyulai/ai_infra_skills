# PR #27149: Fix incorrect string formatting in barrier timeout exceptions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27149](https://github.com/vllm-project/vllm/pull/27149) |
| 作者 | hyongtao-code |
| 合并日期 | 2025-10-18 |
| labels | ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

barrier 超时异常用 printf 风格占位符未被插值,异常消息成了元组;改 f-string 修复。

## 问题现象

当 vLLM 分布式 barrier 超时(barrier 等待或 barrier departure 等待超过 `timeout`)时,抛出的 `RuntimeError` 消息不是预期的格式化文本,而是一个未格式化的元组,例如实际显示为 `("Barrier timed out after %f seconds", 30.0)` 这类内容,用户无法直接读到真实超时秒数。触发条件:`vllm/distributed/utils.py` 中 barrier 等待超过 timeout 后 raise 异常的路径。

## 根因

PR 正文披露:Python 的异常构造函数(`RuntimeError(...)`)不会像 logging 那样对 printf 风格参数做 `%` 插值,传入的格式串与参数会原样保留为元组,导致 `RuntimeError was raised with an unformatted tuple`。代码里两处 barrier 超时都用了 `"Barrier ... %f ...", timeout` 的 logging 式写法。

## 修复方案

改 `vllm/distributed/utils.py` 两处 barrier 超时 raise,把 printf 风格参数替换为 f-string 插值(同时把一处 `%f s` 统一为 `.2f seconds`):

```diff
-                raise RuntimeError("Barrier timed out after %f seconds", timeout)
+                raise RuntimeError(f"Barrier timed out after {timeout:.2f} seconds")
```

```diff
-                raise RuntimeError("Barrier departure timed out after %f s", timeout)
+                raise RuntimeError(
+                    f"Barrier departure timed out after {timeout:.2f} seconds"
+                )
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 `vllm/distributed/utils.py` 中 barrier 与 barrier departure 两个超时异常的消息文本,不影响任何控制流或超时判定逻辑;唯一行为变化是异常消息内容(由元组变为正确格式化文本,且精度显示从 6 位小数变为 2 位)。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
