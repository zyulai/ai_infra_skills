# PR #46463: fix(security): prevent infinite loop in split_audio with NaN audio sa…

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46463](https://github.com/vllm-project/vllm/pull/46463) |
| 作者 | jperezdealgaba |
| 合并日期 | 2026-06-23 |
| labels | ready, multi-modality |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
音频全为 NaN 时 find_split_point 返回 0、split_audio 光标不前进死循环挂死 worker;回退 start_idx 并加前进守卫。(pack 中标题截断)

## 问题现象
当音频的所有采样值都是 NaN 时,`split_audio` 永不前进其循环光标,导致 worker 线程永久挂起(死循环)。正文说明机制:IEEE 754 下 NaN 的比较恒为 False。触发条件为喂入含全 NaN 采样的音频;从标题看属安全加固类修复(恶意/异常输入可造成挂死)。

## 根因
正文披露:`find_split_point` 在全部音频采样为 NaN 时返回 0——因为 NaN 比较恒为 False,记录最静音位置的 `quietest_idx` 保持初值 0;`split_audio` 拿到不前进的切分点后循环光标永远不动,worker 线程被永久挂起。

## 修复方案
改动 `vllm/multimodal/audio.py` 三处思路:
1. `find_split_point`:`quietest_idx` 初值由 `0` 改为 `start_idx`(安全回退),并在能量比较时跳过 NaN:`if not math.isnan(energy) and energy < min_energy:`;
2. `split_audio`:新增前进守卫,切分点未前进时回退到硬切块边界。关键 diff 摘录:

```python
# Guarantee forward progress: if split_point didn't advance,
# fall back to the hard chunk boundary.
if split_point <= i:
    split_point = min(i + chunk_size, audio_data.shape[-1])
```

3. `tests/multimodal/test_audio.py` 新增两个测试:`test_find_split_point_nan_input`(全 NaN 输入必须返回 start_idx 而非 0)与 `test_split_audio_nan_input_terminates`(31 秒全 NaN 音频、max_clip_duration_s=30 必须正常终止、至少切出 2 块且覆盖全部采样)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:音频按能量找切分点的逻辑;对正常音频,守卫仅在切分点不前进时触发、`quietest_idx` 初值变化不影响已有比较逻辑;全 NaN 或含 NaN 能量值的音频由「挂死」变为按硬边界切块。切分位置的质量(最静音点)在含 NaN 片段时可能选择不同位置。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
