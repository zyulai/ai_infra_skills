# PR #57071: [Bugfix][ROCm] AMD-Quark mixed-precision DeepSeek-V4.1 support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57071](https://github.com/vllm-project/vllm/pull/57071) |
| 作者 | xiao-llm |
| 合并日期 | 2026-09-26 |
| labels | bug, rocm, ready, deepseek, quantization, DSv4, DSv4.1 |
| 热度 | 👍 0 · 💬 23 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
`DeepSeek-V4.1-Flash-MXFP4`(MXFP4 专家 + 226 层 MXFP8 attention)加载失败;修四个独立 bug:配置抢占丢 spec、缺 MXFP8 scheme、VL 映射不可见、engram scale 名不匹配。

## 问题现象
`DeepSeek-V4.1-Flash-MXFP4` 完全无法加载。它是 mixed-precision AMD-Quark 导出:专家为 MXFP4(`global_quant_config` = fp4/per_group/group_size=32/e8m0),但 226 条 per-layer attention spec 为 MXFP8(`fp8_e4m3`/`per_block`/`block_size=[32,32]`/`float8_e8m0fnu`)。具体表现:所有 226 条 per-layer spec 被静默丢弃;`NotImplementedError: No quark compatible scheme was found.`;attention 分片静默回退到全局 MXFP4;engram scale 张量永不路由。

## 根因
正文披露四个独立 bug,每个都源于假设 DeepSeek **V4** 的 `128x128` block 几何或 DeepSeek 原生 scale 命名:
1. `DeepseekV4FP8Config.override_quantization_method` 抢占任何全局 scheme 为 MXFP4 的 Quark 导出,而其 `from_config` 把配置重写为单一全局 FP8 scheme(`weight_block_size=[128,128]`),丢弃全部 per-layer spec;`get_quant_method` 的 MXFP8 路径又以 `weight_block_size == [32, 32]` 为门,重写后不可达。
2. `QuarkConfig._is_fp8_w8a8` 只接受 `[128,128]` 的 per-block FP8,OCP-MX 匹配器只接受 1-D `per_group`,32×32 per-block e8m0 权重两头不匹配,dispatch 直接 raise。
3. `configure_quant_config` 在 `__init__` 之前从**类**上读 `hf_to_vllm_mapper`/`packed_modules_mapping`,而 `DeepseekV41ForCausalLM`(VL 变体)只定义为实例属性,`layers.N.attn.wq_a` 永远匹配不到运行时的 `language_model.model.layers.N.attn.wq_a`。
4. Quark 把 engram scale 拼作 `engram.embed.weight_scale`,mapper 规则只匹配字面 `\.scale$` 后缀,永不命中。

## 修复方案
1. `override_quantization_method` 对一切 `quant_method=="quark"` 返回 None,交还 `QuarkConfig`(V4 与 V4.1 两份 quant_config.py)。
2. `QuarkOCP_MX` 新增 unpacked MXFP8 分支:`mxfp8_e4m3` 进 `_WEIGHT_QUANT_KEY_MAP`/`_ACTIVATION_QUANT_KEY_MAP`,新 normalizer `ocp_mx_weight_dtype_and_rows()` 归一两种配置方言(1-D per_group R=1 与 2-D per_block R=32),`packed_factor` 三分(mxfp8=1/mxfp4=2/mxfp6=8/6);loader 用 `loaded_weight.view(torch.uint8).repeat_interleave(rows, dim=0)` 把每 R 行一条的 scale 展开为逐行;kernel 复用 `init_mxfp8_linear_kernel()`(compressed-tensors/INC 同款);顺带修 `get_quant_method()` 与 `get_quant_method_target()` 对 MXFP8 层 dispatch 不一致的潜在问题。MoE 路径显式 raise 并说明 MXFP8 专家不支持(仅 linear 路径)。
3. VL wrapper 把两个映射提升为类属性,前缀规则经模块级常量 `_VL_PREFIX_MAPPING` 共享。
4. mapper 增加 `(engram\.embed)\.weight_scale$ -> \1_tokens.weight_scale_inv` 规则。
关键 diff 摘录(vllm/models/deepseek_v41/quant_config.py):
```python
+        # ``from_config`` below rewrites the config into a single global FP8
+        # scheme, which would discard per-layer specs — so never claim Quark.
+        if isinstance(hf_quant_cfg, dict) and (
+            hf_quant_cfg.get("quant_method") == "quark"
+        ):
+            return None
```

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响 ROCm/AMD 路径(`vllm/models/deepseek_v41/amd/`)的 Quark 量化加载。
- 回归面:`quark/quark.py`、`quark_moe.py`、`schemes/quark_ocp_mx.py`、`ocp_mx_utils.py`、`config_utils.py`、deepseek_v4/v41 quant_config、amd model/vl_model。既有 128×128 per-block FP8 仍走 `QuarkW8A8Fp8PerBlock`(新测试断言)。**范围刻意限定**:nvidia/ 孪生路径存在相同的第 3、4 项问题但本 PR 不动,仅 AMD 路径经硬件验证。E2E:4×MI355X(gfx950/CDNA4)TP=4,48 shards/475 GB 全部加载 74.5s,模型 coherent;GSM8K 8-shot MXFP4 92.34%(flexible)/92.42%(strict),Base FP8 92.87%/92.95%。0 新增代码失败。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
