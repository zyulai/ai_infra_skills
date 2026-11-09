# PR #6677: Revert "fix communicator for non-dp lm head (#6662)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6677](https://github.com/sgl-project/sglang/pull/6677) |
| 作者 | zhyncs |
| 合并日期 | 2025-05-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #6662:撤销把模型输出 scatter 模式按 enable_dp_lm_head 区分的改动,恢复输入输出统一 TP_ATTN_FULL。

## 问题现象
正文仅一句 "This reverts commit a3d7f4b67333a85b9e18ad7f06abe352ce5582db.",Motivation/Modifications 均为空模板,未描述具体现象。被回退的原 PR 标题为 "fix communicator for non-dp lm head (#6662)",本 PR 将其撤销;具体症状与复现条件正文未给出。

## 根因
正文未披露根因。(由 diff 推断)机制:#6662 曾把 `ScatterMode` 中统一的 `model_input_output()` 拆为两个静态方法——输入仍返回 `TP_ATTN_FULL`,而新增的 `model_output_mode()` 按 `global_server_args_dict["enable_dp_lm_head"]` 区分(开启返回 `TP_ATTN_FULL`,否则返回 `FULL`),并让最后一层的 `_compute_layer_output_mode` 与 `qwen2_moe.py` 的 TBO 输入模式使用新方法;本回退删除 `model_output_mode()`,把上述调用点全部恢复为 `model_input_output()`,即输入与输出 scatter 模式重新统一为 `TP_ATTN_FULL`。

## 修复方案
revert commit a3d7f4b,两个文件:
- `python/sglang/srt/layers/communicator.py`:删除 `model_output_mode()` 及 `model_input_mode()`,`_compute_layer_input_mode`、`_compute_layer_output_mode`(最后一层分支)均改回 `ScatterMode.model_input_output()`。
- `python/sglang/srt/models/qwen2_moe.py`:`model_forward_maybe_tbo` 的 `input_data_scatter_mode` 改回 `ScatterMode.model_input_output()`。

关键 diff:
```diff
-    def model_output_mode():
-        """The scatter mode for model output data"""
-        if global_server_args_dict["enable_dp_lm_head"]:
-            return ScatterMode.TP_ATTN_FULL
-        else:
-            return ScatterMode.FULL
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #6662 合入后至本回退落地之间。
- 回归面:`communicator.py` 的层间 scatter 模式计算与 qwen2_moe 的 TBO 前向路径。非 DP LM head(未开 `enable_dp_lm_head`)场景下,模型输出 scatter 模式从 `FULL` 回到 `TP_ATTN_FULL`,数据布局/通信行为随之改变;`enable_dp_lm_head` 开启场景行为与 #6662 前一致。#6662 原本针对的 non-dp lm head 通信问题可能复现,需另行修复。

## 关联
- [#6662](https://github.com/sgl-project/sglang/pull/6662)(标题提到的被回退 PR,正文注明被回退 commit a3d7f4b67333a85b9e18ad7f06abe352ce5582db)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
