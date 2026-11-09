# PR #20463: [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20463](https://github.com/sgl-project/sglang/pull/20463) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-03-14 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
glm4v_moe/glm_ocr 的 load_weights 缺少 Conv3d→Linear 权重拷贝调用,视觉编码器输出乱码;补上调用修复。

## 问题现象
GLM-4.6V 系列模型(glm4v_moe、glm_ocr)出现视觉回归:模型对图像的回答与图片内容完全无关,输出与图像无关的文本。原因是 `Glm4vVisionPatchEmbed` 中 `Conv3d` 被替换为 `Linear` 后,`Linear` 层保留随机初始化权重,视觉编码器产生垃圾 embedding。受影响的是走这两个模型文件的 GLM-4.6V / GLM-OCR 视觉推理。

## 根因
正文披露:PR #20033 在 `Glm4vVisionPatchEmbed` 中把 `Conv3d` 替换为 `Linear`,并在 `glm4v.py` 的 `load_weights()` 末尾添加了 `copy_conv3d_weight_to_linear()` 调用做权重转换;但 `glm4v_moe.py` 和 `glm_ocr.py` 各自有自己的 `load_weights()` 覆盖实现,该调用没有加进去,导致 `Linear` 层持有随机权重。作者通过 `git bisect` 对 625 个提交(v0.5.9 → 7a1ca538)定位确认了这一根因。

## 修复方案
在 `glm4v_moe.py` 与 `glm_ocr.py` 两个文件的 `load_weights()` 末尾补上缺失的转换调用,且仅在非 nextn(非投机解码 draft)路径执行:

```diff
+        if not is_nextn:
+            self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

涉及文件:`python/sglang/srt/models/glm4v_moe.py`、`python/sglang/srt/models/glm_ocr.py`。

## 影响与回归风险
- 影响窗口:PR #20033 合并之后、本修复之前的版本(正文提及以 git bisect 定位,区间起点 v0.5.9 之后引入回归的提交)。
- 回归面:改动只在两个模型 `load_weights()` 末尾追加一次权重拷贝调用,`is_nextn=True` 的投机解码权重加载路径被显式跳过。对已正确加载的 glm4v.py 路径无影响;修复后 glm4v_moe/glm_ocr 权重加载多一步转换,属一次性开销。

## 关联
- Fixes #20462
- PR #20033

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
