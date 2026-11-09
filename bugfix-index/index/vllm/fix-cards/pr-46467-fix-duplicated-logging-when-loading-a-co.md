# PR #46467: Fix duplicated logging when loading a corrupt or partial video

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46467](https://github.com/vllm-project/vllm/pull/46467) |
| 作者 | hhhhhhhhhhhhhhhhho |
| 合并日期 | 2026-06-24 |
| labels | ready, multi-modality |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
损坏/不完整视频加载时 OpenCVVideoBackend 每个坏帧都打 WARNING,重复刷屏;把逐帧日志降级为 debug,保留汇总行为 warning,纯日志级别改动。

## 问题现象
加载损坏或部分损坏(partial)的视频时,`OpenCVVideoBackend` 会对每个损坏帧打一条 WARNING,与既有的单条汇总日志重复,导致日志被逐帧警告刷屏。正文未描述更严重的功能性影响(如输出错误数据),该 PR 定位为纯日志冗长性(verbosity-only)问题。

## 根因
PR 正文披露:`_read_frames_with_recovery` 与 `_read_frames_no_recovery` 中对每个失败帧(grab 失败、retrieve 失败、无法恢复)都调用 `logger.warning`,与最终的单条汇总警告重复。功能逻辑本身(跳帧、恢复)没有问题,只是日志级别用错。

## 修复方案
修改 `vllm/multimodal/video.py`,把 5 处逐帧日志从 `logger.warning` 降级为 `logger.debug`,汇总日志保持 warning 级别:

```diff
-                    logger.warning(
+                    logger.debug(
                         "Failed to grab frame %d during video loading.",
                         idx,
                     )
```

覆盖位置:`_read_frames_with_recovery` 中 grab 失败、retrieve 失败、剩余不可恢复帧(共 3 处);`_read_frames_no_recovery` 中 grab 失败与 retrieve 失败(共 2 处)。另将一处注释 "log warning" 改为 "skip it" 以匹配实际行为。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是使用 OpenCV video backend 加载损坏/不完整视频的多模态场景。
- 回归面:仅日志输出级别变化,行为零影响;运维侧若依赖 WARNING 级别的逐帧信息需改开 debug 级别。正文未附测试计划与测试结果(checklist 中标注 "not required")。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
