# PR #21083: [Perf] Cuda Kernel for Per Token Group Quant

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-07-22 |
| 分类 | Quantization |
| Label | performance, ready, ci/build, deepseek |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 6 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/21083 |

## 一句话结论

把 per-token-group FP8 量化从 Triton 换成 CUDA kernel(带 shared memory 缓存与列 major/UE8M0 scale 支持),DeepGEMM 场景吞吐 +6% 左右。

## 动机

`per_token_group_quant_fp8` 是 DeepSeek 类 block-quant FP8 模型(配合 DeepGEMM)每层都跑的热点算子,此前只有 Triton 实现。PR 改写自 SGLang 的 `per_token_group_quant_8bit.cu`,并额外增加 row-major E8M0、float32 scale tensor 支持,用 shared memory 优化访存。

## 优化点

新增 CUDA kernel `per_token_group_quant_8bit_kernel`:16 线程负责一个 group(128 元素),第一遍 `vectorize_with_alignment` 把 group 数据从 global 读入 shared memory 的同时累加 absmax,`GroupReduceMax` 用 4 步 `__shfl_xor_sync`(mask=0xffff)在 16 线程内做 warp 级 max 归约,得到 `y_s = absmax/448`(UE8M0 模式下再 `exp2f(ceilf(log2f(...)))` 上取到 2 的幂);scale 写出支持 row-major 与 column-major(后者按 pack 重算 stride,为 DeepGEMM 转置 scale 布局)。第二遍从 shared memory 读回、向量化量化写 global,避免原来"先算 absmax 再量化"对输入的两遍 DRAM 读。host 侧按 num_groups 整除性选 groups_per_block(1/2/4/8/16)决定 block 尺寸与 shared memory 用量。Python 侧 `fp8_utils.per_token_group_quant_fp8` 在 CUDA 且 contiguous 时优先走 `torch.ops._C.per_token_group_fp8_quant`,Triton 作为 fallback 保留,`use_ue8m0` 默认值改为 `is_blackwell_deep_gemm_used()` 并统一传入两条路径。

## 关键改动

```text
csrc/quantization/fp8/per_token_group_quant.cu: 新增 per-token-group FP8 量化 CUDA kernel(shared memory 缓存 + warp 归约)
vllm/model_executor/layers/quantization/utils/fp8_utils.py: CUDA kernel 优先路径,Triton fallback
csrc/torch_bindings.cpp: 注册 ops.per_token_group_fp8_quant
tests/kernels/quantization/test_per_token_group_quant.py: CUDA vs Triton 参考实现的数值一致性测试
```

## 性能声明

B200(`VLLM_USE_DEEP_GEMM=1`,Qwen3-30B-A3B-FP8,`vllm bench throughput --input-len 1000 --output-len 100 --enable-expert-parallel`,enforce_eager 经 lm_eval 参数可见):27.00 req/s / 29643.76 total tokens/s / 2700.48 output tokens/s,main 为 25.35 / 27828.75 / 2535.13。gsm8k 5-shot 精度与 main 持平(flexible-extract 0.8469 vs 0.8476,strict-match 均 0.8855)。DeepSeek-R1 EP+DP8(`VLLM_ALL2ALL_BACKEND=deepep_high_throughput`,random ISL 256/OSL 100,1000 prompts):request throughput 25.79 vs 24.59 req/s,output token throughput 2579.29 vs 2458.55 tok/s,mean TTFT 14384.65 vs 14935.81 ms。H100(同 Qwen3-30B-A3B-FP8 throughput 命令):23.88 req/s / 26216.81 total tokens/s vs main 22.35 / 24537.01。

## 验证与回归风险

- 新增 `test_per_token_group_quant.py`:按 shape/column_major/scale_ue8m0/group_size(64/128)参数化,用 mock 使 CUDA 平台判定为 False 走 Triton 作参考,断言量化值(atol/rtol=0.15)与 scale(0.01)一致。
- 风险:kernel 目前只 dispatch `Float8_e4m3fn` 输出(int8 组未接);shared memory 用量为 groups_per_block*group_size*sizeof(T),大 group_size 组合可能超限;`is_column_major` 由 stride 关系推断,非标准 scale 布局可能误判;Triton fallback 仍服务于非 contiguous 输入。

## 关联

- 前序:#19233(同作者的 INT8 量化向量化,提供 vectorization_utils.cuh)、#21476(INT8 per-token-group quant kernel,后续)。
- 后续使用方:#27883(fused blockwise quant RMS norm 复用 per_token_group_quant_fp8)、#20645(flashinfer blockscale fp8 MoE 调用该量化)、#24342(per-token-group FP8 扩展)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
