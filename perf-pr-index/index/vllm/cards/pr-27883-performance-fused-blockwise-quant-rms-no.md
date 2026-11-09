# PR #27883: [Performance] Fused blockwise quant RMS norm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ElizaWszola |
| 合并 | 2025-12-07 |
| 分类 | Quantization |
| Label | performance, ready, torch.compile, nvidia |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 显示 8 个文件因过长被截断,其余见 PR |
| 链接 | https://github.com/vllm-project/vllm/pull/27883 |

## 一句话结论

新增 CUDA kernel 把 RMSNorm 与输出按 block/group 的 FP8 量化融合为一个算子,microbenchmark 上比 unfused 路径快约 30-45%。

## 动机

blockwise FP8(如 DeepGEMM 风格 128-group 量化)模型在每层 RMSNorm 后都要单独跑一次 `per_token_group_quant_fp8`,norm 输出需要完整写回 global memory 再读回量化,产生两次额外的显存往返。PR 提供 "CUDA kernel and fusion code for Fused Groupwise FP8-Quantized RMS Norm",当 `enable_fusion==True` 时把 RMSNorm + FP8 量化合成一个 kernel。

## 优化点

新增 `rms_norm_per_block_quant` CUDA 算子:每个 token 一个 block,先向量化计算 RMS,再按 group(64 或 128,编译期常量分发)计算每组 absmax scale——组内用 `threads_per_group` 协作扫描 + shared memory 归约(新增 `warpReduceMaxSpecialized` 按有效元素数做定制的 warp-level max,替代整块 cub BlockReduce),最后 `norm_and_quant` 阶段直接按 `scale_idx = token * num_groups + group` 取对应组 scale 完成量化写出的同时支持 scale 转置布局(`is_scale_transposed`,配合 DeepGEMM 需要的转置 scale)。int8 时把 scale 取倒数推迟到 `norm_and_quant` 内做,避免多线程共享 scale 元素的额外同步。同时为 torch.compile 提供 fusion pass 集成(标签含 torch.compile,PR 在 `tests/compile/test_fusion.py` 验证 fusion 生成)。原 per-token 路径的 dispatch 也改用 `VLLM_DISPATCH_BOOL` 模板化 has_residual。

## 关键改动

```text
csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu: 新增 rms_norm_per_block_quant_kernel 与 dispatch
csrc/quantization/fused_kernels/layernorm_utils.cuh: groupwise absmax 归约与 norm_and_quant 的 group scale 索引
csrc/dispatch_utils.h: 新增 VLLM_DISPATCH_BOOL / VLLM_DISPATCH_GROUP_SIZE
csrc/torch_bindings.cpp: 注册 ops.rms_norm_per_block_quant
tests/kernels/core/test_fused_quant_layernorm.py: GROUP_SIZES=[None,[1,64],[1,128]] 的 opcheck 测试
```

## 性能声明

PR 附 H100 microbenchmark(`benchmarks/fused_kernels/layernorm_rms_benchmarks.py`,unfused_groupwise_fp8_impl vs fused_groupwise_fp8_impl):unfused 约 32-40、fused 约 22-27(表格未标注单位;覆盖 N=1/2/4/8 tokens、hidden_size 1024-7168、bf16/fp32、有/无 residual、group size 64/128)。端到端测试用 Qwen3-30B-A3B-FP8 + `enable_fusion=True` 离线推理,仅验证 fused kernel 被生成,未报端到端数字。

## 验证与回归风险

- 新增/依赖测试:`tests/kernels/core/test_fused_quant_layernorm.py`(kernel 正确性,含 group size 64/128 与转置 scale)、`tests/compile/test_fusion.py`(fusion pass),均在 `VLLM_USE_DEEP_GEMM=1/0` 两种配置下验证。
- 风险:kernel 仅支持 group_size 64/128(TORCH_CHECK 硬限制);groupwise 分支要求 hidden_size 能被 group_size 整除且线程布局依赖 `blockDim.x / num_groups` 整除,非常规 hidden_size 可能走不通;改动同时触及原 per-token kernel 的 norm_and_quant 签名(标量 scale 变指针),int8 语义需靠既有测试回归。

## 关联

- 同区域前序:#21476(int8 per-token-group quant CUDA kernel)、#24342(FP8 per-token-group quantization 扩展)、#18844(scaled FP8 quant 提速)。
- 上游依赖 DeepGEMM(`VLLM_USE_DEEP_GEMM`)场景下的 norm→quant 链路。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
