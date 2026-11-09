# PR #10375: fix(internvl): fix accuracy issue of normalization

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10375](https://github.com/sgl-project/sglang/pull/10375) |
| 作者 | KEVINTUAN12 |
| 合并日期 | 2025-09-15 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

InternVL 图像预处理用每张图自身的 mean/std 归一化,方差小时 bf16 下过度截断造成精度损失;改回用 ImageNet 固定 mean/std。

## 问题现象

作者在测试 internvl3 8b 时发现精度问题,而 internvl 1b 和 38b 上没有问题,即问题集中在特定规模/配置的模型上。触发条件:走 `InternVLImageProcessor` 的图像/视频预处理路径。修复前后对比(照抄原文):

- acc in main: internvl3 8b mmmu : 0.549
- acc in current branch: internvl3 8b mmmu : 0.55
- MMMU Benchmark time: main 354 s → current branch 353 s

## 根因

正文披露:此前改用每张输入图像自身的 mean 和 variance 做归一化,当时的理由是固定 mean/variance 可能与实际输入图像相差较大、造成归一化精度损失;但用图像自身的 mean/variance,在方差小时归一化结果会偏小,在 bf16 模式下可能因过度截断引发精度问题。因此改回使用 ImageNet 数据集的 mean/variance 做归一化。

## 修复方案

改 `python/sglang/srt/multimodal/processors/internvl.py`:

- 新增 `IMAGENET_MEAN = [0.485, 0.456, 0.406]`、`IMAGENET_STD = [0.229, 0.224, 0.225]` 常量,并用 `lru_cache` 缓存构造好的 mean/std tensor(`_get_normalize_tensors`),避免重复创建;
- `load_video`(视频帧路径)与 `process_mm_data_async`(图像路径)中,删除按图计算 mean/std 的逻辑,统一改用固定的 ImageNet mean/std:

```diff
-            # Using the mean and variance of the ImageNet dataset for all input images can lead to accuracy issues, while using the mean and variance of each input image is a more accurate choice.
-            mean = img.mean(dim=[1, 2], keepdim=True)
-            # Prevent division by zero; clamp to minimum value of 1e-6
-            std = img.std(dim=[1, 2], keepdim=True).clamp(min=1e-6)
             img = (img - mean) / std
```

## 影响与回归风险

- 影响窗口:正文未说明(该 per-image 归一化为此前某次改动引入,具体 PR 正文未指明)。
- 回归面:仅 InternVL 图像/视频预处理的归一化数值路径(`internvl.py` 的 `load_video` 与 `process_mm_data_async`);受影响模型为 `InternVLChatModel` 与 `InternS1ForConditionalGeneration`。方差极小的纯色图不再有除零 clamp 保护,改用固定 std(0.229 等)后不存在除零问题;评测显示 mmmu 略升(0.549→0.55)、耗时基本持平。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
