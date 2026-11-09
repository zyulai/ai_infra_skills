# PR #23060: [fix] Fix dynamic chunking profiling crash on GLM-5 models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23060](https://github.com/sgl-project/sglang/pull/23060) |
| 作者 | Baichuan7 |
| 合并日期 | 2026-04-23 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 21 |
| 分类 | Model-specific Fast Path |
| bug 类型 | crash |

## 一句话结论
GLM-5 开启 dynamic chunking 时 profiling 阶段抛 `AttributeError: _is_extend_in_batch`,修复为在 profiling 前显式设置该标志。

## 问题现象
使用 `--enable-dynamic-chunking` 并配合 GLM-5 模型(带 DeepEP)时,profiling 阶段崩溃,报 `AttributeError: _is_extend_in_batch`。该崩溃的连带影响是:dynamic chunking 在模型上被静默禁用,同时在非首个 PP rank 上引起 KV cache 内存泄漏和 NCCL 超时。修复前状态:dynamic chunking 静默失效并伴随 KV cache 泄漏与 NCCL 超时;修复后:profiling 正常完成,dynamic chunking 按预期工作。本 PR 修复 issue #23057。

## 根因
PR 正文披露:profiling 路径绕过了 `prepare_mlp_sync_batch`,而正常调度路径下正是这个函数负责设置 `_is_extend_in_batch` 的值;因此 profiling 期间该属性未被设置,访问时抛出 `AttributeError`。

## 修复方案
改动位于 `python/sglang/srt/managers/scheduler_pp_mixin.py` 的 `profile_and_init_predictor`:
- 从 `sglang.srt` 相关模块新增导入 `set_is_extend_in_batch`;
- 在 profiling 循环内、调用 `model_runner.forward()` 之前,依据 batch 的 forward mode 显式设置该标志,即 `set_is_extend_in_batch(batch.forward_mode.is_extend())`。

关键 diff 摘录:

```diff
@@ -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):
                 model_worker_batch = batch.get_model_worker_batch()
 
                 forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
+                set_is_extend_in_batch(batch.forward_mode.is_extend())
+
                 _ = model_runner.forward(
                     forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                 )
```

注:PR 正文的 Modifications 一节表述为「调用 `set_is_extend_in_batch(True)`」,实际合并 diff 传入的是 `batch.forward_mode.is_extend()`,以 diff 为准。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:改动仅触及 scheduler 的 PP profiling 路径(`profile_and_init_predictor`),即启用 dynamic chunking 场景下 profiling 阶段的 forward 调用前逻辑。新增的是一处全局标志的显式设置,会使 `_is_extend_in_batch` 在 profiling 期间按 batch 的 extend/decode 属性被赋值;依赖该全局状态的其他读取路径在 profiling 后会看到这一新值(由 diff 推断)。正常服务调度路径未改动。

## 关联
- [issue #23057](https://github.com/sgl-project/sglang/issues/23057)(正文 "Fixes #23057")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
