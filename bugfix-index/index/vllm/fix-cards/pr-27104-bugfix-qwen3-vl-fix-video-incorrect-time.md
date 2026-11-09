# PR #27104: [bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27104](https://github.com/vllm-project/vllm/pull/27104) |
| 作者 | wulipc |
| 合并日期 | 2025-10-17 |
| labels | ready, qwen |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Qwen3-VL 抽帧时 `sampled_fps` 覆写了 `video_fps` 变量,导致时间戳计算错误;改为写入独立变量修复。

## 问题现象
Qwen3-VL 处理视频且 `do_sample_frames=True` 时,时间戳计算不正确。正文对该现象的描述很简短:「The current code overwrites `video_fps` with `sampled_fps`, which causes incorrect timestamp calculations」,未给出具体复现命令、日志或错误输出示例;影响面为 Qwen3-VL 视频输入在开启抽帧采样时的 `_get_video_second_idx` 时间戳/秒索引计算路径。

## 根因
正文披露:当前代码在 `do_sample_frames` 分支里用 `video_fps = sampled_fps if sampled_fps else video_processor.fps` 把采样后的 fps 直接写进 `video_fps` 变量,造成变量被覆写,后续时间戳计算错误。
(由 diff 推断)该分支的语义本是:`video_fps` 保持原始视频 fps(`metadata["fps"]` 也是原始 fps),采样 fps 是另一个量;计算 `num_frames = int(total_num_frames / metadata["fps"] * video_fps)` 时本应乘「采样 fps」,而后续(按秒索引)的时间戳换算仍依赖未被污染的 `video_fps`。覆写后两个用途混在一个变量上,其中一方必然取错值——修复即把采样 fps 写回 `sampled_fps` 自身,`num_frames` 公式改用 `sampled_fps`,`video_fps` 保持原值不再被动过。

## 修复方案
改 `vllm/model_executor/models/qwen3_vl.py` 的 `_get_video_second_idx`,`do_sample_frames` 分支两行(全文改动仅此 2 行):
```python
# here video_fps is the fps of the sampled video, and
# metadata["fps"] refers to the fps of the original video.
sampled_fps = sampled_fps if sampled_fps else video_processor.fps
total_num_frames = metadata["total_num_frames"]
num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)
```
即:不再 `video_fps = ...` 覆写,改写 `sampled_fps`;`num_frames` 计算乘数同步改为 `sampled_fps`;其余逻辑(含 `min_frames`/`max_frames` 截断)不变。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:仅 Qwen3-VL `_get_video_second_idx` 的 `do_sample_frames=True` 路径:`num_frames` 计算值不变(乘的是同一个有效采样 fps);变化在于该函数后续使用 `video_fps` 的位置现在拿到的是原始视频 fps 而非被覆写的采样 fps,时间戳/秒索引随之变化为正确值;`do_sample_frames=False` 路径与其他模型不受影响。正文未附评测或前后对比数据。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
