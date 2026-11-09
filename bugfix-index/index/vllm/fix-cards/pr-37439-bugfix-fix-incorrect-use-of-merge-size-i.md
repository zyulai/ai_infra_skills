# PR #37439: [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37439](https://github.com/vllm-project/vllm/pull/37439) |
| 作者 | cnyvfang |
| 合并日期 | 2026-03-18 |
| labels | bug, qwen, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

Qwen3-VL 视频时间戳误用 `merge_size`(空间参数)分组,非默认 `temporal_patch_size` 下断言失败;改用 `temporal_patch_size` 修复。

## 问题现象

Qwen3-VL 与 Qwen3.5 上,当 checkpoint 的 `temporal_patch_size` 被改为非默认值(如 16,需同步改 `config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`processor_config.json`)后,推理包含视频的输入会在 `get_video_repl` 处直接抛出运行时错误:

```text
AssertionError: timestamps and tokens_per_frame must have the same length
```

报错栈经 `input_preprocessor.preprocess` → `_apply_hf_processor` → `qwen3_vl.py` 的 `_call_hf_processor` → `get_video_repl` 断言。默认配置(`merge_size == temporal_patch_size == 2`)下不触发,因此只在非默认配置/改过的 checkpoint 上暴露。

## 根因

PR 正文披露:`_get_video_second_idx` 把 `video_processor.merge_size` 传给 `_calculate_timestamps` 用于 padding 帧索引和对时间戳分组,但 `merge_size` 在 Qwen3-VL/Qwen3.5 中只作用于**空间**维度,时间维 token 不会被 merger 合并;正确参数应是 `temporal_patch_size`,因为时间分组数与时间戳分组逻辑都取决于 temporal patching 而非空间合并。旧代码在默认配置下"碰巧"正确,只因默认值里 `merge_size` 与 `temporal_patch_size` 数值相等;一旦非默认,`timestamps`(按 merge_size 分组)与 `tokens_per_frame`(按 temporal_patch_size 分组)长度不一致,断言失败。

## 修复方案

改 `vllm/model_executor/models/qwen3_vl.py` 的 `_get_video_second_idx`:取 `video_processor.temporal_patch_size` 并传给 `_calculate_timestamps`:

```diff
-        merge_size = video_processor.merge_size
+        temporal_patch_size = video_processor.temporal_patch_size
         indices = metadata["frames_indices"]
@@ 
-        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
+        timestamps = self._calculate_timestamps(indices, video_fps, temporal_patch_size)
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围;受影响面为 Qwen3-VL 与 Qwen3.5(正文注明该 bug 同时影响 Qwen3.5)。
- 回归面:仅 Qwen3-VL 系视频时间戳计算路径;对默认配置(`merge_size == temporal_patch_size`)行为完全不变,非默认配置下 `len(timestamps) == len(tokens_per_frame)` 恢复成立。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
