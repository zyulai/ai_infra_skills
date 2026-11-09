# PR #10112: fix: fp8 quantization failure of qwen 2.5 VL 7B model

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10112](https://github.com/sgl-project/sglang/pull/10112) |
| 作者 | PanJason |
| 合并日期 | 2025-09-27 |
| labels | ready-to-merge, run-ci |
| 热度 | 👍 7 · 💬 9 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
qwen2.5-VL-7B 的 mlp intermediate size 非 8 对齐致 fp8 量化加载失败;初始化对齐到 8 并对权重零填充修复。

## 问题现象
qwen 2.5 VL 7B 模型的 fp8 量化失败,issue #6828 报告失败发生在视觉模型(visual model)部分。修复后 `w8a8_fp8` 与 `fp8` 两种量化路径均可正常工作。

## 根因
正文修改条目披露:qwen 2.5 VL 7B 的(视觉塔)mlp intermediate size 不是 8 对齐的,而量化路径按 8 对齐假设加载/切分权重,导致 fp8 量化在该模型上失败;作者选择不做内核改动(区别于此前的 #7448 方案),而是在模型初始化与权重加载两侧处理非对齐情况。

## 修复方案
- `python/sglang/srt/models/qwen2_5_vl.py`:视觉塔 `mlp_hidden_size` 初始化改为 8 对齐:`((vision_config.intermediate_size + 7) // 8) * 8`;
- 新增 `python/sglang/srt/layers/utils.py::pad_or_narrow_weight`:当 `start_idx + shard_size` 超出权重维度时,切出有效部分并补零到 shard_size(全越界时返回全零),否则正常 narrow;
- `python/sglang/srt/layers/linear.py` 的 `weight_loader` 与 `weight_loader_v2`、`python/sglang/srt/layers/parameter.py` 的 `load_merged_column_weight`/`load_row_parallel_weight`:加载非 8 对齐权重时改用零填充逻辑;
- `python/sglang/srt/layers/quantization/w8a8_int8.py` `apply`:把 `x_q`/`x_scale` view 成 2D 再调用 `int8_scaled_mm`,输出 reshape 回原形状,修复 `w8a8_int8` 的运行时崩溃。

关键摘录:
```diff
-        mlp_hidden_size: int = vision_config.intermediate_size
+        mlp_hidden_size: int = ((vision_config.intermediate_size + 7) // 8) * 8
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动触及所有走 `weight_loader`/`weight_loader_v2` 与 column/row 并行权重加载的模型,但仅当 `end_idx > 权重维度`(即非对齐特殊情形)才走新增零填充分支,常规对齐权重仍走原 narrow 路径;`w8a8_int8.apply` 的 2D 化改写影响该量化方法的全部调用形状(输出经 view 还原)。正文另指出两个不属本 PR 的遗留问题:#10623(并发请求 rotary_embedding 崩溃)、#10626(qwen2.5 7B instruct 基座模型在 w8a8_int8 下输出乱码)。

## 关联
- https://github.com/sgl-project/sglang/issues/6828
- https://github.com/sgl-project/sglang/pull/7448
- https://github.com/sgl-project/sglang/issues/10623
- https://github.com/sgl-project/sglang/issues/10626

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
