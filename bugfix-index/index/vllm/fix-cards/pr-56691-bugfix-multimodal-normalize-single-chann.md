# PR #56691: [Bugfix][Multimodal] Normalize single-channel audio to 1D

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56691](https://github.com/vllm-project/vllm/pull/56691) |
| 作者 | Woolgathererer |
| 合并日期 | 2026-10-04 |
| labels | bug, ready, multi-modality |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
带显式声道维的单声道音频未被压成 1D,Whisper 预处理报错;normalize_audio 在 mono 目标下返回唯一声道。

## 问题现象
带显式声道维的单声道(mono)音频在 Whisper 预处理中失败:16 kHz 下传入 `(1, 16000)` 波形,经 offline 音频 tuple 路径会抛出 `ValueError: Failed to apply WhisperProcessor`;同样样本用 `(16000,)` 一维数组则正常。副作用是 `AudioProcessorItems.get_audio_length()` 会把长度报成 1 而不是 16000。不需要重采样即可复现。

## 根因
正文披露:`normalize_audio` 在输入声道数已等于目标声道数时提前返回(early return),对 mono 输入留下了 2D 数组——即 `(1, 16000)` 布局不会被展平为 1D,后续 WhisperProcessor 无法处理。

## 修复方案
修改 `vllm/multimodal/audio.py` 的 `normalize_audio`:当声道数已达目标且目标是单声道时,返回唯一声道作为 1D 波形:

```diff
     # No reduction needed if already at target
     if num_channels == spec.target_channels:
-        return audio
+        return audio[0] if num_channels == 1 else audio
```

同时扩展既有 NumPy、Torch 与 parser 测试:mono 透传测试参数化覆盖 `(3,)`、`(1, 3)`、`(3, 1)`、`(1, 1)` 四种布局,parser 测试覆盖 `(16000,)`、`(1, 16000)`、`(16000, 1)` 并新增长度与采样值断言。

测试结果:聚焦回归用例修复前 8 failed / 3 passed,修复后 11 passed;Whisper 预处理对比同一秒 WAV 的三种布局,修复后三者产出完全一致的 feature(shape `(1, 80, 3000)`)、prompt tokens 与 placeholder ranges,parsed audio length 为 16000。作者注明模型生成与精度评估未运行,本地缺 torchaudio 构建故过滤了相关用例。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号。
- 回归面:`normalize_audio` 的所有调用方(音频预处理与 MultiModalDataParser 的音频解析);行为变化仅限「输入为单声道且目标为 mono」分支, stereo 透传(`target_channels=None` 保留 2D 立体声)与多声道降混路径不变;若下游有依赖「透传时原样返回 2D 单声道数组」的逻辑会受影响(由 diff 推断)。

## 关联
- #47693(调整归一化与重采样顺序,其当前补丁未改动该 early return)
- #56023(处理不支持的多声道缩减,同上)
- #51022(优化 downmixing,同上)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
