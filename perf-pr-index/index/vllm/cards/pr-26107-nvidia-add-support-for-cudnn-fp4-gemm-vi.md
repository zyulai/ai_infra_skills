# PR #26107: [NVIDIA] Add support for cudnn fp4 gemm via flashinfer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kaixih |
| 合并 | 2025-10-15 |
| 分类 | Quantization(tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/26107 |

## 一句话结论

通过 FlashInfer 接入 cuDNN FP4 GEMM 作为 NVFP4 线性层可选后端,DeepSeek-R1-FP4-v2 TP8 下端到端约 +1%。

## 动机

NVFP4(modelopt_fp4 / compressed-tensors W4A4)线性层此前只有 trtllm/cutlass/fbgemm/marlin 几种 GEMM 后端,cuDNN 的 FP4 GEMM(需要 `nvidia-cudnn-cu12` 与 `nvidia-cudnn-frontend` 依赖)尚未接入。该算子主要服务于 shared expert 与 output projection 层(当前仅 nvidia/DeepSeek-R1-FP4-v2 使用),cuDNN 实现在这些形状上有更低延迟的潜力。同时旧的 `VLLM_USE_TRTLLM_FP4_GEMM` 布尔开关只会在 cutlass 与 trtllm 之间二选一,后端选择逻辑分散且不易扩展。

## 优化点

机制是后端选择统一化 + 新增 cudnn 分支:把 `VLLM_USE_TRTLLM_FP4_GEMM` 布尔变量替换为 `VLLM_NVFP4_GEMM_BACKEND` 枚举(`flashinfer-cudnn` / `flashinfer-trtllm` / `flashinfer-cutlass`,留空自动选择:有 flashinfer 则 flashinfer-cutlass,否则原生 cutlass,modelopt 路径再退 marlin)。`ModelOptNvFp4LinearMethod` 与 `CompressedTensorsW4A4Fp4` 的 `apply` 里按后缀统一调用 `flashinfer_scaled_fp4_mm(*mm_args, backend=<cudnn|trtllm|cutlass>)`,cuDNN 由 FlashInfer 的该接口透传。选择无可用后端时报错并提示检查平台能力。该 env 变量同时纳入 `compute_hash` 以保证编译缓存一致。

## 关键改动

```text
vllm/envs.py: VLLM_USE_TRTLLM_FP4_GEMM 替换为 VLLM_NVFP4_GEMM_BACKEND(带 choices)
vllm/model_executor/layers/quantization/modelopt.py: NVFP4 线性层后端选择与 cudnn 分支
vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py: 同样的后端选择统一
```

```python
if self.backend.startswith("flashinfer-"):
    backend_name = self.backend[len("flashinfer-"):]
    out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
```

注:PR 正文给出的启用方式是 `VLLM_USE_CUDNN_FP4_GEMM=1`,而 diff 实现为 `VLLM_NVFP4_GEMM_BACKEND` 选择变量,两者记录如上,以 diff 为准。

## 性能声明

nvidia/DeepSeek-R1-FP4-v2,TP8(GPU 型号未注明),`vllm bench throughput`,random 数据集 1024/1024、2048 prompts、max-num-seqs 1024:cutlass 基线 5.09 req/s / 10426.68 total tok/s / 5213.34 output tok/s → cuDNN 5.13 req/s / 10510.00 total tok/s / 5255.00 output tok/s(约 **1% 端到端提升**,初步延迟 benchmark 结论)。GSM8K 5-shot(limit 1319):flexible-extract 0.9484 → 0.9462,strict-match 0.9462 → 0.9454。

## 验证与回归风险

lm_eval gsm8k 精度对比(见上,差异在 stderr 范围内);throughput benchmark A/B。风险点:cuDNN 后端需额外安装 cudnn 依赖且由 env 显式开启;`VLLM_USE_TRTLLM_FP4_GEMM` 被移除,依赖旧变量的脚本会失效;fbgemm 分支仅存在于 compressed-tensors scheme;错误信息从"请用 Blackwell 以上平台"改为通用提示,平台能力校验弱化(由后端探测兜底)。

## 关联

同分类(NVFP4):#25609(fbgemm NVFP4 支持dense models)、#46963(NVFP4 KV cache 于 SM8x/SM12x);均为 FP4 推理栈的不同环节。
