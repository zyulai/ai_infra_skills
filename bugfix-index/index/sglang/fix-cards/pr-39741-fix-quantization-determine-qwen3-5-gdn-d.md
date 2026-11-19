# PR #39741: [Fix] quantization: Determine Qwen3.5 GDN dense vs. GPTQ init from checkpoint precision and dynamic exclusions (#38854)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39741](https://github.com/sgl-project/sglang/pull/39741) |
| 作者 | Aravind0403 |
| 合并日期 | 2026-11-16 |
| labels | quant, run-ci |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

Qwen3.5/3.6/3Next GDN 混合精度 GPTQ checkpoint(如 `XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1`)中 `in_proj_b`/`in_proj_a` 是未量化的 BF16 `.weight` 且 `quantization_config.dynamic` 漏了排除规则,SGLang 仍按 GPTQ/Marlin 构建并在 `gptq_marlin_repack` 处因 `size_n = 96 is not divisible by tile_n_size = 64` 崩溃;修复改为以 checkpoint 张量清单 + dynamic 排除为准判定 dense/quantized 初始化,彻底移除按形状猜测的回退。

## 问题现象

Fixes #38854。两类失败:

- GDN 融合投影 `in_proj_ba`(`in_proj_b` + `in_proj_a`,在 `num_v_heads = 48` 时融合宽度 N = 96)在 safetensors 里留的是未量化 BF16 `.weight`,而 `config.json` 的 `quantization_config.dynamic` 没有 `in_proj_a`/`in_proj_b` 的排除项。此前 SGLang 一律按 GPTQ/Marlin 构建 `in_proj_ba`,丢掉 BF16 `.weight` 张量,在 `gptq_marlin_repack` 报 `size_n = 96 is not divisible by tile_n_size = 64` 而中止加载。
- 反向问题:若仅凭层形状推断 dense(`size_n % 64 != 0`)或依赖部分 fused-shard 排除,当量化分片(`.qweight`/`.scales`)同时存在时会静默分配未初始化的 BF16 权重(例如 64-head 的 `in_proj_ba` 在 TP=4 下,或 `-:.*in_proj_z$` 这类部分排除)。

## 根因

正文披露:dense vs quantized 的判定缺少 checkpoint 依据,依赖形状启发式(`ba_out_per_part % 64 != 0`)和 dynamic 排除规则的不完整匹配。具体包括:`get_dynamic_override` 只按融合名整体匹配、不看分片(`in_proj_qkvz` = `in_proj_qkv`+`in_proj_z`,`in_proj_ba` = `in_proj_b`+`in_proj_a`),导致部分排除与全局默认值组合时判断错误;3-bit GPTQ 的 `pack_factor` 用整数除法 `32 // weight_bits` 算错(`32/3`);CPU 路径 `MergedColumnParallelLinear._load_fused_module_from_checkpoint` 在 `packed_dim == output_dim` 时未用 `adjust_shard_indexes_for_packing` 校正 `pad_output_sizes`,造成 CPU `qzeros` 8 倍零填充损坏;Marlin 不兼容的分块形状此前静默回退 BF16。

## 修复方案

1. **以 checkpoint 为准的 dense/quantized 判定(无形状回退)**:
   - `python/sglang/srt/model_loader/weight_utils.py` 新增 `inspect_safetensors_tensor_names`,在 `get_quant_config` 时把 checkpoint 张量名挂到 GPTQ 系配置上(支持本地 index/分片、缓存 HF 工件、Hub index/header 检查,遵循 `HF_HUB_OFFLINE` 与 `LoadFormat.DUMMY`)。
   - `python/sglang/srt/layers/quantization/gptq/gptq.py` 新增 `_CheckpointTensorManifest`(O(1) `__deepcopy__`、`apply_weight_name_mapper`、前缀规范化)与 `_GPTQCheckpointMixin`,横跨 `GPTQConfig`、`GPTQMarlinConfig`、`CPUGPTQConfig`、`GPTQAscendConfig`;dense 的 `UnquantizedLinearMethod` 只在 `dynamic` 排除或 checkpoint 元数据显示 `.weight` 而无 `.qweight` 时选择,混合 dense/quantized 分片直接拒绝。
   - `python/sglang/srt/layers/quantization/moe_wna16.py` 为 `linear_quant_method == "gptq"` 增加缓存 `_gptq_config` 委托(`set_checkpoint_tensor_names`、`apply_weight_name_mapper`、`update_packed_modules_mapping`、`is_linear_unquantized`、`uses_checkpoint_tensor_manifest`,AWQ 不启用),使 `--quantization moe_wna16` 同样支持 dense GDN 投影与动态排除。
   - `marlin_utils.py`/`gptq.py` 新增 `verify_marlin_supports_layer`,量化层分块形状不兼容 Marlin 时显式报错,不再静默回退 BF16。
   - 从 `qwen3_5.py`、`qwen3_next.py` 移除 `ba_out_per_part % 64 != 0` 启发式。
2. **融合分片 `get_dynamic_override` 规范化、3-bit `Fraction` 与加载期精度校验**:`utils.py` 的 `get_dynamic_override` 经 `resolve_dynamic_fused_shards` 检查所有融合分片,仅当融合名或全部分片被排除才返回 `False`,部分排除或冲突配置抛 `ValueError`;3-bit 保留 `Fraction(32, config.weight_bits)`;新增 `validate_quant_param_compatibility` 接入 `load_weights`,checkpoint/参数精度不匹配时报错(PP 安全)。
3. **GDN 与 CPU 打包权重加载修复**:`Qwen3_5GatedDeltaNet`/`Qwen3GatedDeltaNet._bind_packed_weight_loaders` 改为绑定 GPTQ 参数(`qweight`、`qzeros`、`scales`、`g_idx`),3-bit GPTQ 的 `Fraction` 切分取整,非输出分片参数(如 `g_idx`)干净地走 `_make_packed_weight_loader`;`linear.py` 在 `_is_cpu` 且 `packed_dim == output_dim` 时用 `adjust_shard_indexes_for_packing` 校正 `pad_output_sizes`。
4. 新增 `test/registered/unit/models/test_qwen3_5_gdn_gptq_unquant_fallback.py` 覆盖上述全部行为。

关键 diff 摘录:

```diff
-    config.pack_factor = 32 // config.weight_bits  # packed into int32
+    config.pack_factor = (
+        Fraction(32, config.weight_bits)
+        if config.get_name() == "gptq"
+        else 32 // config.weight_bits
+    )  # packed into int32
```

```diff
+def resolve_dynamic_fused_shards(
+    config: Optional[QuantizationConfig], proj_name: str
+) -> Optional[List[str]]:
+    packed_mapping = getattr(config, "packed_modules_mapping", None)
+    if isinstance(packed_mapping, Mapping) and proj_name in packed_mapping:
+        candidate = packed_mapping[proj_name]
+        if isinstance(candidate, (list, tuple)):
+            return list(candidate)
+    shards = _DYNAMIC_FUSED_SHARDS.get(proj_name)
+    return list(shards) if shards is not None else None
```

```diff
+            pad_output_sizes = output_sizes
+            if (
+                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
+                and param.packed_dim == param.output_dim
+            ):
+                pad_output_sizes = [
+                    param.adjust_shard_indexes_for_packing(
+                        shard_size=size, shard_offset=0
+                    )[0]
+                    for size in output_sizes
+                ]
```

## 影响与回归风险

- 影响窗口:GPTQ 系量化(`gptq`、`gptq_marlin`、`moe_wna16` 的 gptq 分支,含 CPU/Ascend/XPU 变体)加载 Qwen3.5/3.6/3Next 类混合精度 checkpoint 的路径;修复后这类 checkpoint 可正常初始化,此前要么崩溃要么静默错载。
- 行为变化面较大:① 部分排除/冲突 dynamic 配置从静默接受变为 `ValueError`;② Marlin 不兼容形状从静默回退变为报错(报错信息提示在 `quantization_config.dynamic` 中排除该层);③ checkpoint 同时含同层 `.weight` 与 `.qweight` 时报错;④ `packed_modules_mapping` 可从 quantization_config 传入 GPTQ 配置。已按旧规则正确配置的部署不受影响。
- CI 状态:Base 通过;Extra 与 AMD ROCm 10 失败(失败原因 pack 未提供)。Checklist 各项未在正文勾选,准确率/速度基准未提供。
- 正文提及 `GPTQMarlinConfig.is_linear_unquantized` 保留对基类"用 LinearBase(1,1) 探测会破坏 Marlin tile 对齐"问题的规避,防止把量化层误判为未量化。

## 关联

- Fixes #38854。
- 新增测试:`test/registered/unit/models/test_qwen3_5_gdn_gptq_unquant_fallback.py`。
- 涉及文件:`python/sglang/srt/model_loader/weight_utils.py`、`python/sglang/srt/layers/quantization/gptq/gptq.py`、`python/sglang/srt/layers/quantization/moe_wna16.py`、`python/sglang/srt/layers/quantization/marlin_utils.py`、`python/sglang/srt/layers/quantization/utils.py`、`python/sglang/srt/layers/linear.py`、`python/sglang/srt/models/qwen3_5.py`、`python/sglang/srt/models/qwen3_next.py`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
