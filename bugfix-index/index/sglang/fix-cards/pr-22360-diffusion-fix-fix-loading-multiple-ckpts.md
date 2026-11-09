# PR #22360: [diffusion] fix: fix loading multiple ckpts with different precision for a same module

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22360](https://github.com/sgl-project/sglang/pull/22360) |
| 作者 | mickqian |
| 合并日期 | 2026-04-08 |
| labels | quant, run-ci, diffusion |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

diffusion 模型同名模块多精度 ckpt 同时被加载,最终权重取决于加载顺序致 Sana/mova 一致性抖动;过滤重复变体并加 fail-fast 校验。

## 问题现象

Sana(及其他一些模型,含 mova)的一致性测试出现 flakiness(抖动)。触发条件:transformer 的重复精度变体 safetensors 文件被一起加载,最终权重依赖 loader/streamer 的先后顺序,输出不确定。

## 根因

正文披露:重复的 transformer safetensors 精度变体被同时加载是根因——同一参数名在多个文件中重复,后到的张量覆盖先到的,最终权重取决于文件迭代或 streamer 投递顺序,行为非确定。diff 注释补充了具体形态(由 diff 推断):Diffusers checkpoint 有时会在同一目录同时带 `foo.safetensors` 与 `foo.fp16.safetensors`(含分片变体),或混入 consolidated 与 sharded 混合的 checkpoint。

## 修复方案

1. `transformer_load_utils.py`:新增 `_PRECISION_VARIANT_SUFFIX_RE` 正则识别 `.fp16/.bf16/.fp32` 变体文件名,`resolve_transformer_safetensors_to_load` 在解析后调用 `_filter_duplicate_precision_variant_safetensors`——同一族里存在无后缀 canonical 文件时丢弃精度变体(仅精度族则不动),并记录过滤日志;
2. `weight_utils.py`:新增 `_raise_if_duplicate_safetensors_keys`,在 `safetensors_weights_iterator` 加载前扫描全部文件的 tensor key,发现跨文件重复键直接抛 `ValueError` 拒绝加载(报错含重复键数与示例),不再静默覆盖;
3. 新增单元测试 `test_transformer_quant.py` 覆盖过滤逻辑,并更新 `consistency_threshold.json` 多个用例阈值。

关键 diff 摘录(weight_utils.py 报错信息):

```diff
+    raise ValueError(
+        "Duplicate tensor names detected across safetensors files. Refusing to load "
+        "because final weights would depend on file or streamer ordering. "
+        f"Found {len(duplicate_files_by_key)} duplicate tensor name(s). "
+        f"Examples: {examples}. "
+        "This usually means multiple precision variants or consolidated+sharded "
+        "checkpoints were passed together."
+    )
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:`resolve_transformer_safetensors_to_load` 的文件列表筛选(此前能加载的「canonical+精度变体并存」checkpoint 现在只加载 canonical);`safetensors_weights_iterator` 新增前置校验,包含重复键的 checkpoint 从「静默加载(结果不定)」变为「报错拒载」;`consistency_threshold.json` 多组阈值收紧或放宽属于测试配置调整。

## 关联

正文未列出(单元测试文件注释提及 #22360,即本 PR)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
