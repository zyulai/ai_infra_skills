# PR #40082: Integrate flashinfer b12x MoE and FP4 GEMM kernels for SM120/121

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @meena-at-work |
| 合并 | 2026-05-20 |
| 分类 | MoE (tier: major) |
| Label | ready, nvidia, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 10 个文件(两种 backend、config、oracle、utils 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/40082 |

## 一句话结论

为 SM120/121 接入 FlashInfer b12x MoE 与 FP4 GEMM backend,DGX Spark 上 Qwen3-30B 吞吐 +1.8%~+6.0%。

## 动机

PR 正文 Summary:面向 SM120/SM121 GPU(DGX Spark GB10、RTX Pro 6000 Blackwell)新增两个 FlashInfer b12x backend。b12x fused MoE kernel 把 token dispatch、W1 GEMM、SwiGLU、W2 GEMM 融合为一次调用,BF16 hidden states 可直接传入(activation quantization 在内部融合),用于加速 NVFP4 MoE;b12x dense GEMM 用 CuTe DSL warp-level MMA 与 adaptive tile sizing 改善小 M decode 形状下的 SM 利用率,接入 NVFP4 linear 路径。

## 优化点

① MoE:新增 `FlashInferCuteDSLSM12xExperts`(正文称 experts/flashinfer_cutedsl_moe.py,diff 中为 experts/flashinfer_b12x_moe.py),`process_weights_after_loading` 归一化 FP4 block scales 并把 `a2_gscale` 重置为 1.0(SM12x 对 FC2 输入用动态 per-block quantisation,modelopt 校准值会饱和);`vllm/config/kernel.py` 给 `MoEBackend` Literal 增加 `flashinfer_cutedsl_sm12x`,`vllm/utils/flashinfer.py` 增加 `has_flashinfer_cutedsl_sm12x_moe()`,`oracle/nvfp4.py` 在 SM120 家族自动选该 backend,`flashinfer_fp4_moe.py` 的 weight-prep 适配。② dense GEMM:新增 `FlashInferB12xNvFp4LinearKernel`(`NvFp4LinearKernel` 子类),经 `has_flashinfer_b12x_gemm()`(基于 `Sm120BlockScaledDenseGemmKernel`)可用性检查,在 SM120/SM121 自动选中、不可用时回退 `FLASHINFER_CUTLASS`;给 `VLLM_NVFP4_GEMM_BACKEND` 增加 `flashinfer-b12x` 选项。非 SM12x 硬件 oracle 不选这些 backend,沿用现有路径。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py: 新增 FlashInferCuteDSLSM12xExperts,归一化 FP4 block scales、a2_gscale 重置为 1.0
vllm/model_executor/kernels/linear/nvfp4/flashinfer.py: 新增 FlashInferB12xNvFp4LinearKernel(SM120+ 自动选,回退 FLASHINFER_CUTLASS)
vllm/config/kernel.py、vllm/envs.py、vllm/utils/flashinfer.py: MoEBackend / VLLM_NVFP4_GEMM_BACKEND 选项与 capability 检查
vllm/model_executor/layers/fused_moe/oracle/nvfp4.py、quantization/utils/flashinfer_fp4_moe.py: SM12x backend 自动选择与 weight-prep
tests/kernels/moe/test_flashinfer_b12x_moe.py、tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py: 新增 b12x 测试
```

关键机制代码:

```python
class FlashInferB12xNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's b12x CuTe DSL warp-level MMA kernel (SM120+)."""
    @classmethod
    def is_supported(cls, compute_capability=None) -> tuple[bool, str | None]:
        if current_platform.has_device_capability(120) and has_flashinfer_b12x_gemm():
            return True, None
        return (False, "FlashInfer b12x requires SM120+ and FlashInfer "
                       "with Sm120BlockScaledDenseGemmKernel")
```

## 性能声明

PR 正文 Test plan:Full-model throughput on DGX Spark(SM121, Qwen3-30B-A3B-NVFP4),b12x GEMM vs flashinfer-cutlass 为 **+1.8%(1P)、+6.0%(8P)**。pack 的 "perf claims detected" 标注为 (none detected by regex),上述数字来自 Test plan 的 full-model throughput 条目,未给出更细的硬件/精度/workload 参数。Test plan 另列 `pytest tests/kernels/moe/test_cutedsl_sm12x_moe.py`(SM120/SM121 上 24/24 passing)与 `pytest tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py -k b12x`(SM121 上 36/36 passing)。

## 验证与回归风险

Test plan:上述两个 pytest(24/24、36/36)、DGX Spark 上 Qwen3-30B-A3B-NVFP4 full-model throughput 对比,以及非 SM12x 硬件上 oracle 不选新 backend 的回退。Hardware requirements:SM120/SM121、含 flashinfer-ai/flashinfer#3051/#3066/#3080 的 FlashInfer、`nvidia-cutlass-dsl==4.4.2`(4.5.0 对 SM121 生成坏 PTX)、并需把 `warp/mma.py` 打补丁以接受 `sm_121a`。影响路径:SM120/121 上的 NVFP4 MoE 与 NVFP4 linear;不满足条件时回退 `FLASHINFER_CUTLASS` 或现有路径。

## 关联

外部依赖 flashinfer-ai/flashinfer#3051(b12x dense GEMM)、#3066、#3080(b12x fused MoE,其中 #3080 拆分 SM100 与 SM120/121 的 MoE API,`b12x_fused_moe` 为 SM120/121 入口);正文未引用其他 vLLM PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
