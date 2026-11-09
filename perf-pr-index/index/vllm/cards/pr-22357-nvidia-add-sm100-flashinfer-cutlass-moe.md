# PR #22357: [NVIDIA] Add SM100 Flashinfer Cutlass MoE fp8 backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amirkl94 |
| 合并 | 2025-08-19 |
| 分类 | MoE(tier: major) |
| Label | performance, ready, ci/build |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 可用但另有 2 个文件因超长未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/22357 |

## 一句话结论

在 SM100(Blackwell)上为 Llama4 形态的 per-tensor FP8 MoE 接入 FlashInfer CUTLASS fused kernel,服务基准输出吞吐相对非 FlashInfer 路径 +30.4%。

## 动机

FlashInfer 的 CUTLASS fused MoE 原先只支持 nvfp4 量化(`FlashInferExperts` 硬性 assert `quant_dtype == "nvfp4"`),SM100 上运行 per-tensor FP8 的 Llama4 类 MoE 只能回退通用 Triton fused experts。为让 `VLLM_USE_FLASHINFER_MOE_FP8` 用户在 Blackwell 上吃到 CUTLASS grouped GEMM 的吞吐收益,需要给该 kernel 补 FP8(e4m3)数据类型分支并接入 vLLM 的 modular kernel 框架。

## 优化点

三条线:其一,`FlashInferExperts` 接受 `torch.float8_e4m3fn`:`workspace_shapes` 对 fp8 输出形状改为 `(aq_m, aq_n)`(nvfp4 为 `aq_n*2`),`apply` 中 fp8 的 `quant_scales` 走 `[g1_alphas, a2_gscale, g2_alphas, a1_gscale]` 布局且不传 `input_sf`,权重不再 view 成 long;其二,把 FP8 MoE 接入 modular kernel——`Fp8MoEMethod`/`ModelOptFp8MoEMethod` 从布尔 `flashinfer_moe_enabled` 改为 `flashinfer_moe_backend` 枚举(`FlashinferMoeBackend` 枚举从 modelopt.py 移到 flashinfer_utils.py,由 `get_flashinfer_moe_backend()` 按 `VLLM_FLASHINFER_MOE_BACKEND` 解析 throughput→CUTLASS/latency→TENSORRT_LLM),CUTLASS 时经 `maybe_make_prepare_finalize` 提供 `FlashInferCutlassMoEPrepareAndFinalize`、经 `select_gemm_impl` 提供 `select_cutlass_fp8_gemm_impl` 构造的 `FlashInferExperts`;其三,`process_weights_after_loading` 重排:仅 TRTLLM 后端才做 `rotate_flashinfer_fp8_moe_weights`(fp8 CUTLASS 不再旋转),CUTLASS 路径 `swap_w13_to_w31`(flashinfer 与 vLLM 的激活半区相反)+ `register_moe_scaling_factors`(新增注册 `w2_input_scale_inv`);`apply` 中 CUTLASS 分支断言非 block_quant、非 renormalize、Llama4 custom routing,优先走预构建 `self.fused_experts`,否则即时组装。约束仍是 Llama4 专用(`select_cutlass_fp8_gemm_impl` 内 assert)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py: FlashInferExperts 支持 fp8 quant_dtype 与形状/scale 布局分支
vllm/model_executor/layers/quantization/fp8.py: Fp8MoEMethod 后端枚举化 + CUTLASS modular kernel 接入
vllm/model_executor/layers/quantization/modelopt.py: ModelOptFp8MoEMethod 同步接入,枚举外移
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py: FlashinferMoeBackend、prepare_finalize/gemm_impl 构建、w2_input_scale_inv
tests/kernels/moe/test_flashinfer.py: 新增 per-tensor FP8 对拍测试
```

fp8 scale 布局分支(节选):

```python
if self.quant_dtype == torch.float8_e4m3fn:
    quant_scales = [self.g1_alphas, self.a2_gscale,
                    self.g2_alphas, self.a1_gscale]
    a1q_scale = None  # not passing input_sf in fp8
    fc1_expert_weights = w1
    fc2_expert_weights = w2
```

## 性能声明

服务基准(`benchmarks/benchmark_serving.py`,Llama4 形态 ckpt,640 prompts,max-concurrency 256,random 1024 in/1024 out,ignore-eos,no-stream;GPU 型号正文未写明,测试门槛为 SM100/Blackwell):

| 配置 | 时长 (s) | Req/s | 输出 tok/s | 总 tok/s | Mean TTFT (ms) | Median TTFT | Mean TPOT (ms) | P99 ITL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlashInfer CUTLASS(`VLLM_USE_FLASHINFER_MOE_FP8=1`) | 149.65 | 4.28 | 4379.27 | 8749.28 | 1692.46 | 634.95 | 49.81 | 213.05 |
| 无 flashinfer cutlass kernels | 195.18 | 3.28 | 3357.78 | 6708.46 | 2967.73 | 1114.10 | 64.35 | 373.07 |

即输出吞吐 +30.4%、请求吞吐 +30.5%、Mean TTFT −43.0%、Mean TPOT −22.6%。精度:`lm_eval` gsm8k 5-shot limit 500 batch 200(`VLLM_FLASHINFER_MOE_BACKEND=throughput`,`VLLM_USE_FLASHINFER_MOE_FP8=1`,FLASHINFER attention):flexible-extract 0.926 ± 0.0117,strict-match 0.900 ± 0.0134(未附基线对照)。

## 验证与回归风险

新增 `tests/kernels/moe/test_flashinfer.py`:8 组 (M,N,K) × E=16/topk=1,`fused_experts`(Triton)与 `apply_flashinfer_per_tensor_scale_fp8` 对拍(atol 5.5e-2/rtol 1e-2,含 CUDA graph 外路径);含一个依赖 flashinfer#1472、暂标 skip 的 CUTLASS 对拍用例。回归风险:`Fp8MoEMethod` 构造签名从 `(quant_config, moe)` 改为 `(quant_config, layer)`,`apply` 分支大重排(rocm_aiter/marlin/trtllm/cutlass/默认),`process_weights_after_loading` 的 swap/rotate 时机变化——错误组合(如 block_quant + CUTLASS)由 assert 拦截;仍限 Llama4 custom routing。baseline 数字的环境变量未注明(仅说明"without flashinfer cutlass kernels")。

## 关联

前序: #20037(FlashInfer CUTLASS fused MoE backend,nvfp4)。后续: #20645(SM100 FlashInfer MoE block-scale FP8,索引已有卡)。本批次内相关卡: pr-33174(Mistral Large 3 + FlashInfer TRTLLM MoE)。
