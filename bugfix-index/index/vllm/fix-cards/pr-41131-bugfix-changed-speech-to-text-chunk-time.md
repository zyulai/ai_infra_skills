# PR #41131: [Bugfix] Changed speech to text chunk timestamp to cumulative approach

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41131](https://github.com/vllm-project/vllm/pull/41131) |
| 作者 | TobyB1702 |
| 合并日期 | 2026-07-27 |
| labels | bug, documentation, frontend, ready |
| 热度 | 👍 3 · 💬 18 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论

vLLM whisper 转写超过 30s 的长音频时 chunk 时间戳按名义 30s 计算而累积漂移;改为按实际 chunk 长度累加求起始偏移。

## 问题现象

用 vLLM whisper 转写超过 30 秒的长音频时,时间戳越来越不准,每个 chunk 最多偏 1 秒;误差随 chunk 数累积,10 个 chunk 后可偏约 5 秒。修复关联 issue #32588。作者用新增回归测试 `test_chunk_timestamp_offset.py` 验证:旧 approach 产生 `[0.0, 30.0, 60.0]` 的名义偏移,修复后为 `[0.0, 29.5, 29.5+29.7]` 的实际累积偏移。

## 根因

正文披露:旧做法假设 `split_audio` 恰好在 30s 处切开,但实际上它会在期望值之前 1 秒窗口内搜索切点,可能落在 29.2 或 29.8 等任意位置;而时间戳按 `idx * max_audio_clip_s` 的名义值计算,切点提前导致的偏差逐 chunk 累积。修复改为累积法:每个 chunk 的起始时间在上一个 chunk 结束位置上累加。

## 修复方案

修改 `vllm/entrypoints/speech_to_text/base/serving.py`:`_preprocess_speech_to_text` 计算 `chunk_start_offsets`(首个 0.0,其后逐 chunk 累加 `chunk.shape[-1] / sample_rate`)并作为第三个返回值;`_create_speech_to_text` 用 `chunk_start_offsets[idx]` 替换旧的 `float(idx * chunk_size_in_s)`;`_get_verbose_segments` 中 `TranscriptionSegment` 的 `seek` 改为 `int(start_time)`(seek 位置可向下取整)。同步更新取消/inter-chunk 测试的 mock,新增回归测试,并修正文档中 `overlap_chunk_second` 的语义说明(它是切点搜索窗口大小,而非 chunk 间重叠)。

```diff
+        chunk_start_offsets: list[float] = [0.0]
+
+        for chunk in chunks[:-1]:
+            chunk_start_offsets.append(
+                chunk_start_offsets[-1] + chunk.shape[-1] / self.asr_config.sample_rate
+            )
```

## 影响与回归风险

- 影响窗口:正文未说明(场景即关联 issue #32588 描述的 >30s 长音频转写)。
- 回归面:`_preprocess_speech_to_text` 返回值从二元组变三元组,所有调用方与测试 mock 需同步(本 PR 已更新 cancellation、inter-chunk 两处测试);verbose_json 输出中 `seek` 由 float 变 int;transcription 文档描述同步修正。

## 关联

#32588

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
