# PR #24815: Revert "[NPU] fix profiler on npu"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24815](https://github.com/sgl-project/sglang/pull/24815) |
| 作者 | sglang-npu-bot |
| 合并日期 | 2026-05-09 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #24685:删除 profiler 启动时给 torch profiler 传入的 NPU 专用 experimental_config,恢复通用 profiler 行为。

## 问题现象
正文仅一句 "Reverts sgl-project/sglang#24685",未描述具体现象。被回退的原 PR 标题为 "[NPU] fix profiler on npu",本 PR 将其撤销;具体症状、报错与复现条件正文均未给出。按 diff 涉及面推断(由 diff 推断),受影响场景是 NPU 设备上通过 `start_profile` 启动 torch profiler 的路径。

## 根因
正文未披露根因,仅说明本 PR 是对 #24685 的 revert。(由 diff 推断)机制:#24685 曾在 `scheduler_profiler_mixin.py` 的 `start_profile` 中为 NPU 传入 `torch_npu.profiler._ExperimentalConfig(...)`(Text 导出、ProfilerLevel.Level1、关闭 msprof_tx/aic_metrics/l2_cache/op_attr/data_simplification/record_op_args、gc_detect_threshold=None);本次回退把整个 `experimental_config` 参数删除,恢复不传该参数的通用调用。即原"修复"被判定带来回归而整体撤销。

## 修复方案
revert #24685,唯一改动文件 `python/sglang/srt/managers/scheduler_profiler_mixin.py`:在 `start_profile` 构造 profiler 的参数列表中删除 NPU 分支的 `experimental_config` 实参,profiler 配置回到不携带 experimental config 的状态。

关键 diff:
```diff
-                experimental_config=(
-                    None
-                    if not _is_npu
-                    else torch_npu.profiler._ExperimentalConfig(
-                        export_type=torch_npu.profiler.ExportType.Text,
-                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
-                        msprof_tx=False,
-                        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
-                        l2_cache=False,
-                        op_attr=False,
-                        data_simplification=False,
-                        record_op_args=False,
-                        gc_detect_threshold=None,
-                    )
-                ),
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #24685 合入后至本回退落地之间。
- 回归面:影响 NPU 上调度器 `start_profile` 的 profiler 配置——被回退 PR 引入的 Text 导出、Level1 采样等级等 NPU 专属设置全部移除,profiler 行为回到通用默认配置;非 NPU 平台原本就走 `None` 分支,行为不变。若 NPU profiler 需要 experimental_config 才能正常导出,则该能力随之消失,需后续另行修复。

## 关联
- [sgl-project/sglang#24685](https://github.com/sgl-project/sglang/pull/24685)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
