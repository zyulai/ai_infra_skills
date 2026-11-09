# PR #25917: Revert "[AMD]fix: use CUDA event for targeted draft-to-verify sync in EAGLE overlap (#21940)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25917](https://github.com/sgl-project/sglang/pull/25917) |
| 作者 | hanming-lu |
| 合并日期 | 2026-05-21 |
| labels | run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #21940:其 CUDA event 定向同步实质等价硬编码关闭 overlap plan stream,应改用环境变量。

## 问题现象
正文未描述具体现象。说明被回退的 #21940 是针对 EAGLE overlap 的 AMD 修复,本 PR 认为它「essentially hardcoding `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0`」,用户应直接设置该环境变量即可,无需代码内的定向同步。

## 根因
正文披露:#21940 添加的 CUDA event 定向同步(commit 5141d8ae21f2e446a0022cb7d009d56702d34744)实质上把 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0` 的效果硬编码进代码,与已有的环境变量开关重复;正确做法是让用户自行设置 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0`,因此回退该提交。

## 修复方案
整体 revert,删除两处 CUDA event 同步代码:
- `python/sglang/srt/speculative/eagle_worker_v2.py`:`forward_batch_generation` 中 draft() 之后记录 `_draft_done_event` 的逻辑,以及 `verify()` 中 plan_stream_ctx 内 `self.plan_stream.wait_event(self._draft_done_event)` 的等待逻辑;
- `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`:同样删除对应的 event record 与 wait_event 两段。

## 影响与回归风险
- 影响窗口:#21940 合入后至本 PR 之前(由回退对象推及)。
- 回归面:EAGLE v2(含 multi-layer)投机解码的 draft→verify 流水线同步路径;恢复为原先 plan_stream 的同步行为,此前依赖该 event 修复的 AMD overlap 场景需改用 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0` 环境变量达到同样效果。

## 关联
- sgl-project/sglang#21940(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
