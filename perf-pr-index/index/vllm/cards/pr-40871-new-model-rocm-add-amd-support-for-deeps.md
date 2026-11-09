# PR #40871: [New Model][ROCm] Add AMD support for DeepSeek V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @whx-sjtu |
| 合并 | 2026-05-05 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, performance, new-model, rocm, speculative-decoding, ready, ci/build, v1, tool-calling, deepseek, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 部分可用——因篇幅丢弃了 3 个文件,可见部分含约 17 个文件(CUDA/HIP kernel、MoE 后端、attention、平台注册等) |
| 链接 | https://github.com/vllm-project/vllm/pull/40871 |

## 一句话结论

把 DeepSeek V4(Flash/Pro)移植到 AMD ROCm/MI355X:HIP 化 fused kernel、AITER MLA sparse attention、MoE 后端选择与 ROCm fallback,使 MI355X 上可服务并通过 GSM8K。

## 动机

DeepSeek V4 的 vLLM 实现此前依赖 CUDA 专属路径:融合的 qnorm/RoPE/KV-insert CUDA kernel(PDL 启动)、tilelang 实现的 mhc(manifold hyper-connections)层、FlashMLA sparse 前向、三路 aux stream 并行 GEMM、E8M0 scale 等,均无法在 ROCm 上编译或运行。本 PR 为 AMD(MI355X)补齐这条 Model-specific Fast Path,使 DeepSeek-V4-Flash 与 DeepSeek-V4-Pro 可在 ROCm 上部署。

## 优化点

移植而非新算法,要点如下。CUDA kernel HIP 化:`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` 增加 `hip/hip_fp8.h` 与 `rocm_cvt_float_to_fp8_e4m3`(区分 HIP_FP8_TYPE_OCP / fnuz),warp mask 按 64 线程 warp 调整,ROCm 用标准 launch(无 PDL);`topk_softplus_sqrt` 的 warpReduceSum 改为 THREADS_PER_ROW 参数化 butterfly(兼容 CUDA WARP_SIZE=32 与 ROCm CDNA 64),非 2 次幂专家数(192/320/448/576)在 WARP_SIZE=64 下改用 narrow LDG(ELTS_PER_LDG=1)满足对齐断言,并放开原先 CUDA-only 的 torch op 注册。attention 侧:decode 走 `rocm_forward_decode_fallback`,prefill 走 `rocm_sparse_attn_prefill`,O 投影走 `rocm_inv_rope_einsum` 的 BF16 参考路径,sparse indexer 增加 `rocm_aiter_sparse_attn_indexer_native` 路径(支持 skip_k_cache_insert);因 hang 问题禁用三路 aux stream(串行执行)并去掉 `launch_pdl` 参数。MoE:mxfp4 后端在 ROCm 优先 `AITER_MXFP4_BF16`(含权重/scale shuffle 预处理),DeepseekV4 routing 时优先 TRITON_UNFUSED(AITER 兜底,注明出于精度);`triton_unfused` 注册为合法 moe backend;`deepseek_v4_fp8` 加入 ROCm 平台量化方法列表。数值适配:AITER scaled_mm 与 triton block scaled mm 在 ROCm 上把 E8M0 scale upcast 到 fp32;mhc 三处 tilelang kernel 提供 numerically equivalent 的纯 torch fallback。

## 关键改动

```text
csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu: HIP FP8 转换与 launch 路径
csrc/moe/topk_softplus_sqrt_kernels.cu: THREADS_PER_ROW 参数化 butterfly + narrow LDG,解除 CUDA-only 限制
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: AITER MLA sparse decode/prefill/einsum ROCm 实现(diff 部分截断)
vllm/model_executor/layers/deepseek_v4_attention.py: ROCm 分支(fallback attention、串行 aux stream)
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py: ROCm 后端优先级与 AITER 权重 shuffle
vllm/model_executor/layers/mhc.py: tilelang kernel 的 ROCm torch fallback
vllm/platforms/rocm.py: 注册 deepseek_v4_fp8 量化方法
```

ROCm 侧 kernel 移植(节选):

```cpp
#ifdef USE_ROCM
__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
  #if defined(HIP_FP8_TYPE_OCP)
  __hip_fp8_e4m3 fp8_val(val);
  #else
  __hip_fp8_e4m3_fnuz fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
#endif
```

## 性能声明

PR 正文无性能数字(素材正则未检出性能声明,亦无吞吐/延迟对比)。给出的是功能性验证:MI355X(docker `rocm/vllm-dev:deepseek-v4-mi35x`)上 DeepSeek-V4-Flash(TP4、`triton_unfused` MoE、enforce-eager)与 DeepSeek-V4-Pro(TP8、fp8 KV cache、`VLLM_ROCM_USE_AITER=1`、enforce-eager)均可 launch 服务;GSM8K(8-shot,lm_eval):Flash 0.9439/0.9431(flexible/strict),Pro 0.9538/0.9545。

## 验证与回归风险

验证以上述 launch + GSM8K 为准;`test_topk_softplus_sqrt.py` 的 skipif 从 `is_cuda` 放宽为 `is_cuda_alike` 以覆盖 ROCm。回归风险面较广:CUDA 路径的行为保持依赖若干"别名收敛"设计(narrow LDG 在 WARP_SIZE=32 下塌缩回宽 load、butterfly 参数化在 32 线程下等价),若实现有误会影响 CUDA MoE topk;ROCm 上多个 fast path 被有意降级(aux stream 串行、PDL 关闭、BF16 O 投影参考路径、torch fallback 的 mhc),属性能换稳定;AITER MoE 权重 shuffle 改变权重布局,模型加载路径变更。

## 关联

同区域:DeepSeek V4 的 CUDA fast path(如 #59511 中涉及的 DeepSeek-V4.1-Flash 生态)、mori/ROCm EP 通信(#41751)。PR 正文未引用具体关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
