# PR #27814: [Refactor] Make FP8 Linear Ops use kernel abstraction

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vllmellm |
| 合并 | 2026-01-20 |
| 分类 | Quantization (tier: major) |
| Label | rocm, ready, ci/build, cpu, nvidia, ready-run-all-tests |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files;diff 摘录另有 20 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/27814 |

## 一句话结论

重构 FP8 linear 内核集成:用统一的 `ScaledMMLinearKernel` 抽象替换 `Fp8LinearOp`,FP8/INT8 全部 scaled-MM 后端走同一内核选择路径(重构,无新内核)。

## 动机

此前 FP8 linear 有两套并行机制:旧的 `Fp8LinearOp` 封装(自带 act_quant_static/act_quant_group_shape 参数与 `maybe_create_device_identity` 等散落调用)与 INT8 侧的 `ScaledMMLinearKernel`/`choose_scaled_mm_linear_kernel` 抽象。两条路径的后端选择、量化 key、参数命名各不相同,代码重复且不一致,新增后端(如 FlashInfer)时只能绕开抽象单独接线。PR 引用 #11785 的通用接口设想,目标是 clarity、maintainability、path consistency。

## 优化点

机制上分三层:(1) 抽象层——`ScaledMMLinearKernel` 改为 `Generic[_ConfigT, _ParamsT]`,新增 `FP8ScaledMMLinearKernel` 与 `Int8ScaledMMLinearKernel` 两个中间基类;FP8 基类在 `apply_weights` 中统一完成输入按需量化(`QuantFP8`,static/dynamic 由 `activation_quant_key` 的 scale 描述符决定)再调用子类 `apply_scaled_mm`,参数访问从硬编码的 5 个参数名改为 `layer_param_names` 序列;config 拆为 `FP8ScaledMMLinearLayerConfig`(weight/activation `QuantKey` + out_dtype)与 `Int8ScaledMMLinearLayerConfig`。(2) 选择层——`init_fp8_linear_kernel`/`init_int8_linear_kernel` 按 `QuantKey` 组合在平台优先级表(`_POSSIBLE_FP8_KERNELS`:CUDA 上 FlashInfer → CUTLASS → torch per-tensor/channel-wise;ROCM 上 ROCm skinny-GEMM → torch rowwise/per-tensor/channelwise)里选内核,并支持 `VLLM_DISABLED_KERNELS` 环境变量禁用特定内核(暂直接读 os.environ,TODO 迁入 envs)。(3) 迁移层——`fp8.py`、`fbgemm_fp8.py`、`modelopt.py`、ptpc、compressed-tensors、quark 等 scheme 的 `Fp8LinearOp` 全部替换为 `init_fp8_linear_kernel(...)` + `apply_weights(layer, x, bias)`,删除 `maybe_create_device_identity` 调用;fp8.py 的 activation quant key 选择保留原语义(static → per-tensor;dynamic + cutlass 支持 → per-token;否则 per-tensor dynamic)。

## 关键改动

```text
vllm/model_executor/layers/quantization/kernels/scaled_mm/ScaledMMLinearKernel.py: 泛型基类 + FP8/INT8 中间基类,FP8 基类统一 apply_weights
vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py: 平台优先级内核表 + init_fp8/init_int8_linear_kernel + VLLM_DISABLED_KERNELS
vllm/model_executor/layers/quantization/fp8.py + fbgemm_fp8.py + compressed_tensors/schemes/*: Fp8LinearOp 迁移到新内核
tests/compile/test_silu_mul_quant_fusion.py + tests/kernels/quantization/test_scaled_mm_kernel_selection.py: 按 force_kernel/GroupShape 参数化,新增 TestFP8Layer
```

关键摘录(fp8.py 的 QuantKey 选择):

```python
if self.act_q_static:
    activation_quant_key = kFp8StaticTensorSym
elif cutlass_fp8_supported():
    activation_quant_key = kFp8DynamicTokenSym
else:
    activation_quant_key = kFp8DynamicTensorSym
self.fp8_linear = init_fp8_linear_kernel(
    activation_quant_key=activation_quant_key,
    weight_quant_key=kFp8StaticTensorSym, ...)
```

## 性能声明

无性能吞吐声明;PR 正文只给 lm_eval gsm8k 精度单值(未附修改前基线):CUDA `RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8`(cutlass)flexible 0.5027 / strict 0.5011;ROCm `RedHatAI/Qwen3-8B-FP8-dynamic`(torch row-wise)0.8795 / 0.8734;`Qwen/Qwen3-0.6B`(ptpc_fp8 torch per-tensor)0.3798 / 0.3889;`RedHatAI/Qwen2-72B-Instruct-FP8` torch per-tensor 0.8855 / 0.8287、rocm per-tensor 0.8741 / 0.8143。测试条件(硬件型号、batch)未在正文注明。

## 验证与回归风险

测试面:`test_silu_mul_quant_fusion.py` 改为按 `force_kernel`(ROCm: ROCmFP8/PerTensorTorch;CUDA: FlashInfer/CUTLASS/PerTensorTorch)参数化,验证 fused RMSNorm/quant 与 SiLU+Mul quant 模式;`test_scaled_mm_kernel_selection.py` 迁到 `Int8ScaledMMLinearLayerConfig`/`CPUInt8ScaledMMLinearKernel`/`AiterInt8ScaledMMLinearKernel`;新增 `TestFP8Layer`/`TestBlockFP8Layer` 助手;CI 新增 `.buildkite/lm-eval-harness/configs/models-small-rocm.txt`。回归风险:所有 FP8/INT8 量化 scheme 的 apply 路径(compressed-tensors INT8 的 `input_scale`/`input_zero_point`/`azp_adj` 参数注册方式改变,统一注册 None);ROCm Aiter fusion/quant 路径重写;`init_*` 的内核选择若与旧 dispatch 顺序不一致会静默换内核,可用 `VLLM_DISABLED_KERNELS` 强制回退。Follow-ups:#31818(统一 TestFP8Layer)、#32268、`VLLM_DISABLED_KERNELS` 进 envs。

## 关联

依赖/呼应 #11785(通用 ScaledMMLinearKernel 接口);follow-up #31818、#32269(已完成)、#32268。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
