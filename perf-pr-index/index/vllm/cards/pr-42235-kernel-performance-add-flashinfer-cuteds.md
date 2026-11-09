# PR #42235: [Kernel][Performance] Add FlashInfer cutedsl NVFP4 GEMM backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmangkad |
| 合并 | 2026-06-22 |
| 分类 | Quantization (tier: major) |
| Label | ready, nvidia |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含新增 `FlashInferCuteDslNvFp4LinearKernel` 及 kernel config/测试/collective_fusion 注册 |
| 链接 | https://github.com/vllm-project/vllm/pull/42235 |

## 一句话结论

为 SM10x 添加 FlashInfer cutedsl NVFP4 GEMM 后端并设为 auto 最高优先级,serving 基准 tok/s/user 最高提升 27.07%。

## 动机

NVFP4 dense GEMM 在 SM100/SM103 上已有 cutlass、cudnn、trtllm、b12x 等 FlashInfer 后端选择,但测试显示 cutedsl 后端在宽并发范围内最快,值得作为可选项并在 auto 选择时优先采用。vLLM 的 linear kernel 体系已按平台/后端注册候选列表,新增后端只需实现 kernel 类并插入优先级列表。

## 优化点

新增 `FlashInferCuteDslNvFp4LinearKernel`:仅支持 SM10x(`is_device_capability_family(100)`)且需安装 FlashInfer。权重处理复用 cutlass 路径的 swizzled + padded layout(`swizzle_blockscale`、`pad_nvfp4_weight_for_cutlass`);apply 时调用 `scaled_fp4_quant`(is_sf_swizzled_layout=True, backend="flashinfer-cutedsl")量化激活,pad 后调用 `flashinfer_scaled_fp4_mm(backend="cute-dsl")`,再裁剪 padding 输出。接入三处:`KernelConfig` 合法值与文档加入 `flashinfer_cutedsl`;`_POSSIBLE_NVFP4_KERNELS` 的 CUDA auto 选择列表把它排在首位;另为 collective_fusion 的 all-gather FP4 融合模式注册 `cute-dsl` 后端变体(use_8x4_sf_layout=False,a_scale_view="float8")。

## 关键改动

```text
vllm/model_executor/kernels/linear/nvfp4/flashinfer.py: 新增
  FlashInferCuteDslNvFp4LinearKernel(is_supported/apply_weights)
vllm/model_executor/kernels/linear/__init__.py: 后端映射与 auto 优先级注册
vllm/config/kernel.py: --linear-backend 取值加入 flashinfer_cutedsl
vllm/compilation/passes/fusion/collective_fusion.py: all-gather FP4
  pattern 注册 cute-dsl 变体
tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py 与
tests/models/quantization/test_nvfp4.py: 新后端参数化测试
```

## 性能声明

PR 正文:cutedsl 在 concurrency 1-512 全范围最快,tok/s/user 相比被测的其他 FlashInfer 后端最多提升 27.07%。条件:`nvidia/Llama-3.1-8B-Instruct-NVFP4`,SM103 设备,random 数据集,512 input / 512 output tokens。详细数据以图形式给出(PR 正文未列数字表格)。

## 验证与回归风险

验证:kernel 级 `test_flashinfer_nvfp4_gemm` 参数化加入 "cute-dsl"(非 SM10x 跳过);模型级 `test_nvfp4` 加入 `flashinfer_cutedsl` 后端(含 EAGER/compile 组合)。风险:仅覆盖 dense NVFP4 GEMM,权重布局与 cutlass 一致因此共享 padding 逻辑;若 FlashInfer 未安装或非 SM10x,is_supported 返回 False 回退其他后端;collective_fusion 注册的变体 use_8x4_sf_layout=False,与 trtllm 的 8x4 layout 行为不同,需注意 TP 路径下的 scale layout 匹配。

## 关联

同 Quantization 分类其他 NVFP4 后端:flashinfer_cutlass/cudnn/trtllm/b12x(kernel 注册处与 B12x 邻接,b12x 因上游 CUTLASS SM121 guard 未解决仍需显式 opt-in)。Kernel/性能相关:#27284(CUTLASS FP8 swap-AB)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
