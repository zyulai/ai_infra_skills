# PR #8141: Revert "Fix different device type adjustment in PP"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8141](https://github.com/sgl-project/sglang/pull/8141) |
| 作者 | saienduri |
| 合并日期 | 2025-07-18 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

回滚 #7760:其设备类型调整导致 AMD 性能 CI 失败;恢复 PP 下 pyobj 传输按 `torch.cuda.current_device()`/`force_cpu_device` 的原设备选择。

## 问题现象

AMD 性能 CI 失败。正文说明 #7760 "Causing AMD performance CI failures",并给出日志链接 [actions/runs/16364699908/job/46239248927](https://github.com/sgl-project/sglang/actions/runs/16364699908/job/46239248927)。正文未给出具体失败指标数值。

## 根因

正文披露:失败由 #7760("Fix different device type adjustment in PP")引起,但未披露机制。(由 diff 推断)#7760 曾把 PP(流水线并行)与 TP 通信中 pyobj 序列化传输的设备选择,从硬编码 CUDA 改为基于 `self.device` / `get_device()` 的动态设备(`broadcast_pyobj`/`point_to_point_pyobj` 增加 `device` 形参,`send_object`/`recv_object` 改用 `self.device`);本回滚将其恢复为原先的 `torch.cuda.current_device()` 硬编码与 `force_cpu_device` 开关。

## 修复方案

按 revert 提交恢复 4 个文件:

- `python/sglang/srt/distributed/parallel_state.py`:`send_object`/`recv_object` 中张量设备从 `self.device` 恢复为 `.cuda(device=torch.cuda.current_device())` / `device=torch.cuda.current_device()`。
- `python/sglang/srt/managers/scheduler.py`:`event_loop_pp` 与 `recv_requests` 中对 `point_to_point_pyobj`/`broadcast_pyobj` 的调用删除 `device=self.device` 实参(共 5 处)。
- `python/sglang/srt/managers/tp_worker.py`:删除一处 `device=self.device` 实参。
- `python/sglang/srt/utils.py`:`broadcast_pyobj` 签名从 `device: Optional[str] = None` 恢复为 `force_cpu_device: bool = True`,设备计算恢复为 `torch.device("cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu")`;`point_to_point_pyobj` 删除 `device` 形参,内部张量统一用 `torch.cuda.current_device()`:

```diff
-    device: Optional[str] = None,
+    force_cpu_device: bool = True,
...
-    if device is None:
-        device = get_device()
+    device = torch.device(
+        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
+    )
```

## 影响与回归风险

- 影响窗口:含 #7760 改动的版本受影响(依据正文声明;正文未给出具体版本号)。
- 回归面:改动触及 `broadcast_pyobj`/`point_to_point_pyobj` 与 `GroupCoordinator.send_object/recv_object` 的设备选择,影响 PP/TP 下请求与对象传输路径。恢复后设备选择重新硬编码为 CUDA 语义;#7760 原本针对的「不同设备类型」场景(如非 CUDA 后端显式指定设备)能力回退,AMD 上性能 CI 恢复通过但需留意相关设备适配需求。

## 关联

- [#7760](https://github.com/sgl-project/sglang/pull/7760)(正文声明被回滚的 PR)
- [AMD 性能 CI 失败日志](https://github.com/sgl-project/sglang/actions/runs/16364699908/job/46239248927)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
