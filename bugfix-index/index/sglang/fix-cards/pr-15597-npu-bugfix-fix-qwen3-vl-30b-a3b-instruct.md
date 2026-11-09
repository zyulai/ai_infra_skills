# PR #15597: [NPU][Bugfix] fix Qwen3-VL-30B-A3B-Instruct accuracy loss

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15597](https://github.com/sgl-project/sglang/pull/15597) |
| 作者 | cen121212 |
| 合并日期 | 2025-12-31 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

NPU 上 Qwen3-VL 精度偏低:get_cos_sin_with_position 补上 2 维 positions 的 mRoPE 处理,MMStar 0.598→0.6713。

## 问题现象

NPU 上 Qwen3-VL-30B-A3B-Instruct 在 MMStar 评测(evalscope,1500 题)整体精度明显偏低:修复前 OVERALL mean_acc 0.598(coarse perception 0.648、fine-grained perception 0.504、instance reasoning 0.676、logical reasoning 0.624、math 0.748、science & technology 0.388)。修复后 OVERALL 0.6713,各子项 0.508~0.816,其中 logical reasoning 0.748、math 0.816 提升最大。评测命令与生成参数(top_p 0.8、temperature 0.7 等)见正文。

## 根因

正文披露:需要对 Qwen VL 补上 mRoPE 的处理(「Add the processing for mRoPE for Qwen VL」),修改点是 `rotary_embedding.py` 的 `get_cos_sin_with_position`。机制由 diff 描述:该函数原来只处理 1 维 positions(`index_select` 后 reshape/repeat),而 Qwen VL 的 mRoPE 位置张量是 2 维(含多个位置分量),走 1 维逻辑会得到错误的 cos/sin,导致注意力位置编码错乱、精度下降(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/layers/rotary_embedding.py` 的 `get_cos_sin_with_position`:按 `positions.ndim` 分支,1 维保持原逻辑;2 维走新分支,用 `cos_sin_cache[positions]` 取值后按 `mrope_section` 切分重排(`mrope_interleaved` 时用 `apply_interleaved_rope`),得到 `position_cos/position_sin`:

```diff
+        assert positions.ndim == 1 or positions.ndim == 2
+        if positions.ndim == 1:
             cos_sin = self.cos_sin_cache.index_select(0, positions.flatten())
             ...
+        else:
+            assert self.mrope_section
+            cos_sin = self.cos_sin_cache[positions]
+            ...
+            self.position_cos = cos.repeat(1, 2).view(-1, 1, 1, last_dim).contiguous()
+            self.position_sin = sin.repeat(1, 2).view(-1, 1, 1, last_dim).contiguous()
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:`rotary_embedding.py` 是共用函数,1 维 positions 分支逻辑保持不变;新增 2 维分支只影响使用 mRoPE 的 VL 模型;正文测试平台为 NPU,其他平台未见数据。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
