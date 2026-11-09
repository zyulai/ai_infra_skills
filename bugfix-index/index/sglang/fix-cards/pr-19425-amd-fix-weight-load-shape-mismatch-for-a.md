# PR #19425: [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19425](https://github.com/sgl-project/sglang/pull/19425) |
| 作者 | billishyahao |
| 合并日期 | 2026-02-27 |
| labels | amd, deepseek, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

amd dsr1 0528 MXFP4 权重加载形状不匹配:exclude 规则匹配不上改名后的 MTP 层致选错量化方法,加权重名映射修复。

## 问题现象

加载 [amd/DeepSeek-R1-0528-MXFP4](https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4) 时出现张量形状不匹配(tensor shape mismatch)问题导致加载失败(正文配了报错截图)。

## 根因

正文披露:amd dsr1 0528 新增了正则规则 `re:model.layers.61.*` 用于排除 MTP 层,但该 checkpoint 里 MTP 层已被重命名为 `model.decoder.mlp.experts.*`,导致 exclude 规则匹配不到目标层,选错了 `QuarkFusedMoEMethod`,进而形状不匹配。

## 修复方案

两处配合:

1. `python/sglang/srt/models/deepseek_nextn.py`:`DeepseekV3ForCausalLMNextN` 增加 `hf_to_sglang_mapper = WeightsMapper(orig_to_new_substr={"model.layers.61": "model.decoder"})` 类属性(仿照 Qwen2.5 VL 的实现模式),把 HF 权重名 `model.layers.61*` 映射到 `model.decoder*`;
2. `python/sglang/srt/layers/quantization/quark/quark.py`:`QuarkConfig.__init__` 把 `exclude` 列表提为实例属性 `self.exclude_layers`,新增 `apply_weight_name_mapper()` 方法调用 `hf_to_sglang_mapper.apply_list()` 对 exclude 规则做同样的重命名,`get_quant_method` 改用该实例属性判断是否跳过量化——这样 exclude 正则与改名后的实际层名对齐。

关键 diff 摘录(quark.py):

```diff
+    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
+        self.exclude_layers = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+
     def get_quant_method(
         self, layer: torch.nn.Module, prefix: str
     ) -> Optional["QuantizeMethodBase"]:
-        exclude_layers = cast(list[str], self.quant_config.get("exclude"))
         if should_ignore_layer(
-            prefix, ignore=exclude_layers, fused_mapping=self.packed_modules_mapping
+            prefix,
+            ignore=self.exclude_layers,
+            fused_mapping=self.packed_modules_mapping,
         ):
```

验证:gsm8k 200 题,Accuracy 0.955、Invalid 0.005(EAGLE 投机解码、aiter attention、tp 8)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:改动 Quark 量化配置的 exclude 层判定路径——原 `get_quant_method` 每次从 quant_config 现取 exclude 列表,现改为实例属性且可被 mapper 改写;未配置 `hf_to_sglang_mapper` 的模型不受影响。`DeepseekV3ForCausalLMNextN` 增加类属性仅影响 NextN/MTP 权重加载的名称映射。

## 关联

- [amd/DeepSeek-R1-0528-MXFP4](https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4)(模型仓库,正文引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
