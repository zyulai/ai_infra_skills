# PR #67006: [Quantization] Online weight-only NVFP4 for dense linear layers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stecasta |
| 合并 | 2026-11-16 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, ready, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 vllm/model_executor/layers/quantization/online/nvfp4.py、online/lm_head.py、kernels/linear/__init__.py、kernels/linear/nvfp4/{base,b12x,flashinfer}.py、vllm/config/quantization.py、docs/features/quantization/online.md 及测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/67006 |

## 一句话结论

新增 `nvfp4_weight_only` 在线量化:FP16/BF16 权重在加载期量化为 NVFP4 并跑既有 W4A16 kernel 选择器(无需 NVFP4 checkpoint);主要用于带宽受限 GPU 上的 BF16 spec-decode drafter,Qwen3.8-27B-NVFP4 + DFlash2 在 DGX Spark c=1 解码 +13.1%、RTX PRO 6000 +14.1%。

## 动机

带宽受限 GPU(DGX Spark GB10、RTX PRO 6000)上,BF16 drafter 的权重访存是瓶颈;已有 `nvfp4` 仅作为 `draft_lm_head_quantization` 专用路径(lm_head.py 内私有 `_LMHeadNvfp4Method`),没有面向普通 dense linear 的在线 NVFP4 weight-only 方案。#49060 直接构建 Marlin、早于 `OnlineLinearBase` requantization 契约,非重复实现。

## 优化点

1. `Nvfp4WeightOnlyOnlineLinearMethod`:加载期把 fp16/bf16 权重量化为 packed E2M1 + E4M3 per-1x16-block scale + fp32 global scale,跑平台 W4A16 kernel(SM100/103 默认 FlashInfer CuTe-DSL,SM12.x Marlin,可用 `--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}'` 钉住);激活保持模型 dtype;MoE 不量化。
2. W4A16 kernel 选择纳入激活 dtype:CuTe-DSL 与 b12x 仅 BF16,FP16 改走 Marlin(此前 FP16 在 SM100/103 可能选中 BF16-only kernel);`NvFp4LinearLayerConfig` 增加 `act_dtype`,`can_implement` 检查。
3. 量化优先用 `scaled_fp4_quant`(native),否则 torch packer;两者 scale 完全相同,码字只在 E2M1 精确平局处不同、误差相同。
4. global scale 跨 TP rank 归约(与 #49764 起的 online FP8 一致),量化权重与 TP 大小无关;注意这改变 `draft_lm_head_quantization: "nvfp4"` 在 TP>1 的数值。
5. 选择方式:整模型 `--quantization nvfp4_weight_only`、按层 `quantization_config.targets`、或仅 drafter(`--speculative-config` 中 `"quantization": "nvfp4_weight_only"`)。

## 关键改动

```python
# vllm/model_executor/layers/quantization/online/nvfp4.py
+    def process_weights_after_loading(self, layer: Module) -> None:
+        weight = self.get_weight_for_quantization(layer)
+        amax = amax_for_tp_weight_quant(
+            weight_amax(weight).float().reshape(1), _is_tp_sharded(layer)
+        )
+        packed, scales, global_scale = _quantize_nvfp4(weight, amax)
+        replace_parameter(layer, "weight", packed)
+        replace_parameter(layer, "weight_scale", scales)
+        replace_parameter(layer, "weight_global_scale", global_scale)
+        self.kernel.process_weights_after_loading(layer)
```

```python
# vllm/model_executor/kernels/linear/nvfp4/flashinfer.py(BF16-only 检查)
+    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
+        act_dtype = config.act_dtype if config is not None else None
+        if act_dtype not in (None, torch.bfloat16):
+            return False, f"requires BF16 activations, got {act_dtype}"
+        return True, None
```

## 性能声明

数字均照抄 PR 正文(未独立核实)。Qwen3.8-27B-NVFP4 + DFlash2(k=7),BF16 drafter vs `nvfp4_weight_only` drafter,SPEED-Bench coding,OSL 8K,nightly `f5d93296c8`,decode tok/s:

- DGX Spark (GB10):c=1 +13.1%,c=4 +22.7%*(c=4 两组输出长度不同,读作约 +15 到 20%),c=8 +4.9%;GSM8K-256 0.965 → 0.961。
- RTX PRO 6000 Max-Q:c=1 +14.1%,c=4 +16.2%,c=8 +1.4%;GSM8K-256 0.961 → 0.969。
- GB300:c=1 +0.1%,c=4 −2.4%,c=8 +2.9%;GSM8K-256 0.969 → 0.961。
- AL 在每一行都在噪声内(GB10 c=1:3.838 → 3.934);对照 FP8 per-channel drafter,GB10 c=1 为 +7.6%。
- 整模型 Qwen3-8B:15.27 → 5.99 GiB,GSM8K-256 与 BF16 在三块 GPU、BF16 与 FP16 下均在噪声内。
- TP=2 on 2x GB200:每个 rank 的 global scale 与 TP=1 相等(column/row-parallel 层),强制按 rank 归约时如预期失败。数字取自 `b291bae1e0`。

## 验证与回归风险

- GB10、RTX PRO 6000 Max-Q、GB300 上全部通过:`tests/quantization/test_online.py`、`tests/model_executor/kernels/test_b12x_linear.py`、`tests/quantization/test_quantization_config_args.py`、`tests/v1/worker/test_gpu_draft_lm_head.py`、`tests/v1/e2e/spec_decode/mtp/qwen3_5/test_mtp.py -k draft_lm_head_quantization`。
- 新测试:native vs reference 量化(BF16/FP16、零值、E2M1 平局)、global scale 的 TP shard 不变性、dtype 感知的 W4A16 选择(含 forced BF16-only backend 拒绝 FP16)。
- 回归风险:未覆盖 B200 与 TP>1 时的 AL;TP>1 下旧 `draft_lm_head_quantization: "nvfp4"` 数值改变;`input_size_per_partition` 需为 16 的倍数。

## 关联

非重复:#49060;TP global scale 先例:#49764(online FP8);被替换的 lm_head 私有实现 `_LMHeadNvfp4Method` 移除,`_LM_HEAD_METHODS["nvfp4"]` 改指新方法。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
