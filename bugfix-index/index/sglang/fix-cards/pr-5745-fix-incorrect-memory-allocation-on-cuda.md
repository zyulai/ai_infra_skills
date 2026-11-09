# PR #5745: [Fix] Incorrect Memory Allocation on CUDA:0 by Non-Zero CUDA Processes in TP/DP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5745](https://github.com/sgl-project/sglang/pull/5745) |
| 作者 | yhyang201 |
| 合并日期 | 2025-05-09 |
| labels | high priority, ready-to-merge |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
TP 已修的 CUDA:0 误分配在 DP 下仍出现,疑因 recv_pyobj 反序列化;把多模态 pixel_values 留在 CPU、调度时再搬上本卡修复。

## 问题现象
fix #5732:非零 CUDA 进程在 CUDA:0 上产生不正确的内存分配;该问题已在 TP 场景修复,但在 DP 场景仍会出现。正文描述较简略,未给出具体报错日志或更细的复现步骤。

## 根因
正文仅初步披露:"问题似乎出在调用 `recv_pyobj` 时的反序列化过程中"(原文 "The problem appears to be caused during deserialization when calling `recv_pyobj`",作者当时仍在调查)。机制细节正文未展开;从 diff 看,修复把多模态 pixel_values 在 processor 内先搬回 CPU、在 `prepare_for_extend` 时才搬到当前 device(由 diff 推断:GPU 上的 pixel_values 张量随消息反序列化时会在 CUDA:0 上产生分配)。

## 修复方案
- `managers/multimodal_processors/base_processor.py`:`process_mm_data` 返回前,若 `pixel_values` 是 torch.Tensor 则 `.to("cpu")`。
- `managers/schedule_batch.py`:ScheduleBatch 新增 `multimodal_inputs` 字段;`prepare_for_extend` 收集各请求的 multimodal_inputs,并把每个 mm_item 的 pixel_values `.to(self.device, non_blocking=True)`;`filter_batch`/`merge_batch` 同步裁剪/合并;`get_model_worker_batch` 改用 batch 级 `self.multimodal_inputs` 而非逐请求取。
- `configs/deepseekvl2.py`、`configs/janus_pro.py`:补 `__contains__`(使 `key in config` 按 `__dict__` 成员判断)。

关键 diff:
```diff
+        if "pixel_values" in result and isinstance(
+            result["pixel_values"], torch.Tensor
+        ):
+            result["pixel_values"] = result["pixel_values"].to("cpu")
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:多模态请求的 pixel_values 设备归属与上卡时机、batch filter/merge 对 `multimodal_inputs` 的维护、`get_model_worker_batch` 字段来源;两个 config 类新增 `__contains__` 改变 `in` 操作语义(由 diff 推断:原先会落入默认对象行为)。

## 关联
- 修复目标 issue:#5732

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
