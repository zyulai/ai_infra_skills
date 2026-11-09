# PR #61365: [Bugfix] Fix Gemma 4 encoder CUDA graph capture OOM at startup

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61365](https://github.com/vllm-project/vllm/pull/61365) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-10-17 |
| labels | bug, multi-modality, nvidia, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | resource |

## 一句话结论
Gemma 4 编码器 CUDA graph 捕获缓冲按「最大帧数×最宽行」分配出 6.69 GiB 巨缓冲,32GB 卡启动即 OOM;按 budget×行宽逐图捕获、行数贴合 token 预算修复。

## 问题现象
开启 `cudagraph_mm_encoder` 后,`nvidia/Gemma-4-26B-A4B-NVFP4` 在 32GB GPU(1x RTX 5090)引擎启动阶段 OOM,无需发请求即可复现(issue 原命令,4/4 次失败):

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.69 GiB. GPU 0 has a total capacity of 31.45 GiB of which 4.94 GiB is free.
```

仅图片模式(同命令 `{"image": 1, "video": 0}`)也在编码器 graph 捕获时 OOM(644 MiB 分配失败,8.43 GiB 已在 private pools)。此外旧代码下视频及 70/140 token 图片批次的 graph 输出与 eager 输出不符(视频 cosine 仅 0.185~0.191)。

## 根因
PR 正文披露:Gemma 4 编码器 graph 输入 pixels 形状为 `(rows, patches_per_row, 768)`,每行一张图或一个视频帧;`prepare_encoder_cudagraph_capture_inputs` 在两个维度上都把缓冲做得过大:(1) 行数对所有 budget 都用 `max(max_batch_size, max_frames_per_batch)`,开视频时为 58×32=1856 行,而能塞进 4096 token 的批至多 58 行;(2) 每行宽度取最宽图像 `min(budget, 280)*9=2520` patches,而视频帧或 70-token 图只需 630。于是 4096-token graph 得到 `(1856, 2520, 768)` bf16 缓冲,即 7,184,056,320 字节(issue 中的 6.69 GiB),patch embedder 首个算子还需同尺寸张量。只限行数不够(#61092 的做法):vision tower 仍在约 4 倍于预算的 patches 上运行、graph pool 达 8.57 GiB,下一 budget 预热即 OOM;宽度问题单独也会让仅图片配置失败。

## 修复方案
全部改动在 `vllm/model_executor/models/gemma4_mm.py`(+49/-10),复用 encoder CUDA graph manager 现有 `capture_axes` 支持:

1. `get_encoder_cudagraph_config` 声明一个捕获轴:行宽(soft tokens)`(70, 140, 280)`,上限为配置的最大值——每个 budget×行宽各捕获一张图,捕获缓冲不再超过 `budget * 9` patches。
2. `select_encoder_cudagraph_items` 返回批次行宽作为轴 key(视频帧 70,图片为 padding 后宽度);混合宽度图片按最宽者打包。
3. `prepare_encoder_cudagraph_capture_inputs` 按该宽度建行,行数取 `min(max(max_batch_size, max_frames_per_batch), max(1, budget // width))`(单行下限仅为覆盖行宽超预算的捕获场景):
```diff
-        max_size = max(max_batch_size, max_frames_per_batch)
+        max_size = min(
+            max(max_batch_size, max_frames_per_batch),
+            max(1, token_budget // per_item_output),
+        )
```
4. `get_encoder_cudagraph_item_specs` 为图片设置 `path_output_tokens` 为打包宽度,保证 manager 不会往一张图塞超过其行数的行;`output_tokens` 仍为真实数量用于切片输出。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(验证基线为 commit `fe4df3df0eb703862c686b256e9fe44fa94c8945`,0.31.1rc1.dev296)。
- 回归面:仅 `gemma4_mm.py`,manager 与其他模型不变。权衡:issue 配置下编码器 graph 从 7 张增至 21 张(3 张永不回放,各占一行);已可工作的纯图配置 graph pool 从 0.93 涨到 1.12 GiB、KV cache 从 7.71 降到 7.46 GiB(最终行数调整前测得);混合宽度图片按最宽打包,可能需更大 budget 或回退 eager(输出仍正确)。验证:修复后 3/3 次启动成功,21 张 graph、CUDA graph pool 2.61 GiB,图像/视频请求回答正确,视频 graph 输出对齐 eager(cosine 0.9994);CPU 单测 34 passed,旧代码上新增 4 个用例失败。正文同时列出不在此 PR 范围的既有缺陷(video padding 帧输出偏移、`max_soft_tokens` 560/1120 崩溃、`Gemma4UnifiedForConditionalGeneration` 捕获失败)。

## 关联
- [issue #61045](https://github.com/vllm-project/vllm/issues/61045)(Fixes)
- [#61092](https://github.com/vllm-project/vllm/pull/61092)(对比方案:只限行数,仍 OOM 且视频输出错误、无测试)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
