# PR #11735: [Bugfix] Fix precision error in LLaVA-NeXT feature size calculation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11735](https://github.com/vllm-project/vllm/pull/11735) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-01-04 |
| labels | ready |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

LLaVA-NeXT/OneVision 特征尺寸计算用 float64 与 HF 的 float32 不一致,导致部分图片尺寸输出错误;改用 numpy float32 计算修复。

## 问题现象

对某些特定图片尺寸(如 (198, 176)、(176, 198)),vLLM 的 LLaVA-NeXT 预处理结果与 HuggingFace 不一致。仓库中的处理测试 `test_processor_prompt_replacements` 此前因这些尺寸失败而被注释掉并标注 `FIXME: image_size [(198, 176), (176, 198)]`,即这些宽高组合下特征尺寸计算出错,影响多模态输入占位符替换的正确性。

## 根因

正文披露:vLLM 当前的特征尺寸计算使用 Python float(即 float64),在某些边界情况下会与 HF 的输出产生不一致;作者尝试用 float32 后问题消失(正文原话:"It seems that using float32 fixes the problem (not really sure why)")。即根因是计算精度与 HF 使用的 float32 不匹配,导致 aspect ratio / scale factor 比较与取整结果不同。

## 修复方案

修改 `vllm/model_executor/models/llava_next.py` 与 `llava_onevision.py` 的 `_get_num_unpadded_features`:把 aspect ratio 与 scale factor 的计算显式包成 numpy float32,并加注释 `# NOTE: HF resizes based on float32`。例如:

```diff
-        original_aspect_ratio = original_width / original_height
-        current_aspect_ratio = current_width / current_height
+        # NOTE: HF resizes based on float32
+        original_aspect_ratio = np.array(original_width / original_height,
+                                         dtype=np.float32)
+        current_aspect_ratio = np.array(current_width / current_height,
+                                        dtype=np.float32)
```

scale_factor 同样改为 `np.array(current_width / original_width, dtype=np.float32)`。同时在 `tests/models/decoder_only/vision_language/processing/test_llava_next.py` 中移除 FIXME,把 (198, 176)、(176, 198) 重新加入测试参数。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅涉及 LLaVA-NeXT 与 LLaVA-OneVision 两个模型的图像特征尺寸计算路径,均为纯 Python/numpy 计算,不涉及内核或调度;风险主要是特定图片尺寸下结果与 HF 仍存在细微差异。同时恢复了两个之前被跳过的测试用例。

## 关联

- FIX #11704

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
