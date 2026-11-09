# PR #47972: Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jimmy-adams |
| 合并 | 2026-08-07 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready, ci/build, deepseek, quantization |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/47972 |

## 一句话结论

让 vLLM 在 AMD 上正确加载 DeepSeek-V4 Quark 混合量化 checkpoint(per-block FP8 linear + NVFP4 MoE,NVFP4 走 emulation kernel),GSM8K 精度恢复 99.83%。

## 动机

DeepSeek-V4 的 AMD Quark checkpoint 是混合量化布局:不同部分使用 per-block FP8 linear 与 NVFP4 MoE 专家。此前 `QuarkW8A8Fp8` 只覆盖 per-tensor/per-channel FP8;DeepSeek-V4 专属 fused module 映射也不应注入通用 `QuarkConfig`。

## 优化点

新增 `QuarkW8A8Fp8PerBlock` scheme:`_is_fp8_w8a8` 识别 weight per_block [128,128] + 动态 per_group 激活(group_size 必须等于 block_size[1])的组合,激活 group size 从 `block_size[1]` 推导而非独立硬编码;DeepSeek-V4 AMD 模型路径承接 `gate_up_proj -> [w1, w3]`、`fused_wqa_wkv -> [wq_a, wkv]` 映射与 scale 张量命名差异(`.scale` → `weight_scale_inv`、`.input_scale` → `.input_scale_2`);Quark NVFP4 MoE fused `w13` global scale 对齐既有 ModelOpt/compressed-tensors 约定——w1/w3 global scale 不同时告警并使用 w1 scale,替代此前的 `max(w1, w3)` 策略;NVFP4 MoE backend 候选集合加入 EMULATION。

## 关键改动

```text
vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py: 新增 QuarkW8A8Fp8PerBlock(kFp8Dynamic128Sym 激活 + kFp8Static128BlockSym 权重)
vllm/model_executor/layers/quantization/quark/quark.py: _is_fp8_w8a8 接受 per-block 组合;packed_modules_mapping 从类级默认拷贝(防实例污染)
vllm/model_executor/layers/quantization/quark/quark_moe.py: fused w13 改用 w1 global scale 并告警
vllm/models/deepseek_v4/amd/model.py: checkpoint scale 参数名解析(_resolve_param_name)与模型级 packed_modules_mapping
vllm/model_executor/layers/fused_moe/oracle/nvfp4.py: NVFP4 MoE backend 集合加入 EMULATION
```

## 性能声明

无吞吐数据,仅精度:MI350/355 上 lm-eval 8-shot、1319 题 GSM8K(服务配置 TP8、`--kv-cache-dtype fp8`、`VLLM_ROCM_USE_AITER=1`、cudagraph FULL_DECODE_ONLY):`amd/DeepSeek-V4-Pro-NVFP4` flexible-extract 0.9477 / strict-match 0.9484;`amd/DeepSeek-V4-Flash-NVFP4` 94.84,对比原版 95.00,恢复 99.83%。

## 验证与回归风险

tests/quantization/test_quark.py 新增 per-block 检测/拒绝、block_size 必填、fused mapping 隔离等用例;CI 新增 GFX950 大模型 GSM8K eval(Pro 配置因 850GiB checkpoint 冷 NFS 读超时暂时禁用)。风险:fused w13 scale 策略变化影响所有 Quark NVFP4 MoE 用户(w1/w3 scale 不一致时精度可能受影响,以告警提示);per-block FP8 scheme 路由变化可能改变已有 checkpoint 的量化路径;per-block FP8 的 E2E 覆盖待真实 checkpoint 可用。

## 关联

同分类 Quantization:#53175(Gemma4 FP8 KV 注意力后端选择)。区域相关:DeepSeek-V4 AMD 模型路径、Quark 量化栈。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
