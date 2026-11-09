# PR #27284: [CI] Fix Nemotron nightly mixed precision checkpoints test

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27284](https://github.com/sgl-project/sglang/pull/27284) |
| 作者 | b8zhong |
| 合并日期 | 2026-06-05 |
| labels | (none) |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
模型加载时的在线量化日志代码在混合精度量化下出错,导致所有 Nemotron NVFP4 checkpoint 无法工作;改为 getattr 安全取值并校验 tuple 后仅记录日志。

## 问题现象
当量化为混合精度(mixed precision)时服务无法正常工作,所有 Nemotron NVFP4 checkpoint 都受影响;Nemotron nightly mixed precision checkpoints 测试因此失败(该测试由 #18005 弄坏)。正文未给出具体报错日志。

## 根因
正文披露:由 https://github.com/sgl-project/sglang/pull/18005 引入。`ModelRunner.load_model` 中对 `self.model.quant_config.quantized_layers` 的直接访问在量化是混合精度时会 break。正文并指出:这段代码本应只移到 quark 量化路径,而且它只是一个日志。

## 修复方案
`python/sglang/srt/model_executor/model_runner.py` 的 `load_model`:先通过 `getattr(getattr(self.model, "quant_config", None), "quantized_layers", None)` 安全取值,再仅当 `quantization is not None and isinstance(quantized_layers, tuple) and len(quantized_layers) == 2` 时解包 `layer_types, quantized_layers_count` 并打印 info 日志;不再对 `quant_config`/`quantized_layers` 做多层 hasattr 直取。日志内容不变。

关键 diff:
```diff
+        # TODO: Move this online-quantization reporting out of ModelRunner.
+        quantized_layers = getattr(
+            getattr(self.model, "quant_config", None), "quantized_layers", None
+        )
         if (
-            hasattr(self.model, "quant_config")
-            and hasattr(self.model.quant_config, "quantized_layers")
-            and self.server_args.quantization is not None
+            self.server_args.quantization is not None
+            and isinstance(quantized_layers, tuple)
+            and len(quantized_layers) == 2
         ):
```

## 影响与回归风险
- 影响窗口:正文未说明;损坏自 #18005 合入起。
- 回归面:所有带 `--quantization` 且模型带 `quant_config.quantized_layers` 的加载路径——quantized_layers 为非二元组(如混合精度量化场景)时原先会出错,现在跳过日志;原有能正常打印的场景(quark 等)日志行为保持。仅影响日志分支,不改量化本身。

## 关联
- 弄坏者 PR:#18005

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
