# PR #11772: [Bugfix] Fix LLaVA-NeXT feature size precision error (for real)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11772](https://github.com/vllm-project/vllm/pull/11772) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-01-07 |
| labels | documentation, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
LLaVA-NeXT/OneVision 特征数计算的浮点精度与 HF 不一致,改为全程 float32 运算对齐输出。

## 问题现象
标题披露:LLaVA-NeXT 的特征数量(feature size)存在精度误差;正文仅链接到 PR #11735 的评论区(指向此前一次未修彻底的尝试,标题中的 "(for real)" 亦指此),未以文字描述具体症状。关联测试为 processor prompt replacements 测试:特征数算错会导致图像 token 占位符替换数量与 HF 处理器输出不一致。新增的触发尺寸参数为 (161, 184) 与 (184, 161) 两种图片尺寸。

## 根因
正文未披露根因,仅以 "(for real)" 指向前次修复未成功的讨论。(由 diff 推断)旧实现中 `current_height`/`current_width` 为 Python int,而 aspect ratio、scale_factor 用 `np.array(..., dtype=np.float32)` 包装——除法在 int 域先按 float64 计算再降为 float32,且后续乘法/整除在 int 与 float32 之间混用,舍入顺序与 HF(全程 float32)不一致;特定宽高(如 161x184)下算出的 unpadded features / newline features 偏离 HF 结果。修复把所有中间量统一为 `np.float32` 后再做运算,只在最后转 int,与 HF 输出保持一致。

## 修复方案
修改 `vllm/model_executor/models/llava_next.py` 与 `llava_onevision.py` 的 `_get_num_unpadded_features`:全部输入先转 `np.float32`(`current_height_f`、`original_width_f` 等),aspect ratio、scale_factor、padding、最终 `unpadded_features`/`newline_features` 均在 float32 域计算后取整;OneVision 版本同步处理 `ratio` 分支。测试在 llava_next 与 llava_onevision 的 prompt replacement 测试中补充 (161, 184)、(184, 161) 两个尺寸。

```diff
+        current_height_f = np.float32(npatches * num_patch_height)
+        current_width_f = np.float32(npatches * num_patch_width)
+        original_width_f = np.float32(original_width)
+        original_height_f = np.float32(original_height)
-        original_aspect_ratio = np.array(original_width / original_height,
-                                         dtype=np.float32)
+        original_aspect_ratio = original_width_f / original_height_f
```

## 影响与回归风险
- 影响窗口:正文未说明;标题 "(for real)" 表明此前(如 #11735 相关尝试)已修过一次未修彻底。
- 回归面:`llava_next.py` 与 `llava_onevision.py` 的特征数计算路径,直接影响这两类模型的图像 token 预算与 prompt 占位符替换;纯数值计算重排,不改算法语义。对依赖旧(错误)特征数的用户,修复后 prompt 长度会与 HF 对齐而略有变化。

## 关联
- https://github.com/vllm-project/vllm/pull/11735#issuecomment-2572196154

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
