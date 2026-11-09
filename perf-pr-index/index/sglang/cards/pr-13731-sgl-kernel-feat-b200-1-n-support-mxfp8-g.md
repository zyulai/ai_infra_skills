# PR #13731: [sgl-kernel][Feat][B200][1/N]Support MXFP8 Grouped GEMM in Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HydraQYH |
| 合并 | 2025-12-04 |
| 分类 | MoE(tier: major) |
| Label | quant, sgl-kernel, run-ci |
| Diff | 素材未提供 diffstat(5 个文件因体积原因从 diff 中省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/13731 |

## 一句话结论

为 B200/SM100 新增 CUTLASS MXFP8 block-scaled Grouped GEMM 及配套 Group Quant 内核,量化内核可吃到 B200 HBM 带宽的 85% 以上。

## 动机

MoE 专家 GEMM 需要 grouped 形式的 MXFP8 量化计算,但此前 sgl-kernel 缺少 MXFP8 Grouped GEMM 与配套的分组量化内核。量化要输出的 Scale Factor 布局特殊(cuBLAS 的 D block-scaling factors 布局),TensorRT-LLM 用 STG.8 指令逐字节写 Global Memory,存取效率低,值得一个更快的原生实现。

## 优化点

两块新内核:一是 `es_sm100_mxfp8_blockscaled_grouped_mm`,基于 CUTLASS/CuTe 的 SM100 block-scaled grouped GEMM,支持 BF16/FP16 输出,经 pre-compute kernel 在 device 端为每个专家算好指针偏移、stride 与 SFA/SFB layout 后由 CUTLASS grouped GEMM 启动(启用 PDL);二是 Group Quant 内核 `es_sm100_mxfp8_blockscaled_grouped_quant`,单个 kernel 内完成所有 group 的 MXFP8 量化(warp 内求 32 元素块 max,算 E8M0 scale factor,`rcp.approx.ftz.f32` 快速倒数,输出 FP8 E4M3)。针对 TRT-LLM STG.8 写 scale 的低效,采用三项优化:256bit Load、100% Occupancy(按 occupancy API 计算网格)、scale factor 经 shared memory 用 TMA STORE(`cp_async_bulk`)写 global memory。

## 关键改动

```text
sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu: grouped GEMM 入口,SM100 guard + BF16/FP16 分发
sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh: 分组量化内核(E8M0 SF 计算、TMA STORE 写 SF)
sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_launcher.cuh: device 端 pre-compute(offset/layout/stride)+ CUTLASS grouped GEMM 启动(PDL)
sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_functor.cuh: 各专家指针/stride/layout 的 device functor
sgl-kernel/csrc/common_extension.cc: 注册 es_sm100_mxfp8_blockscaled_grouped_mm / _grouped_quant 两个 op
```

```cpp
// group_quant.cuh:块 max -> E8M0 scale factor,TMA 写出
float sf_val = block_max * reciprocal_approximate_ftz(448.0f);
__nv_fp8_e8m0 tmp_sf_val;
tmp_sf_val.__x = __nv_cvt_float_to_e8m0(sf_val, __NV_SATFINITE, cudaRoundPosInf);
...
if (threadIdx.x == 0) {
  cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
      squeeze_tiled_tensor_sf.data().get(),
      squeeze_tiled_tensor_shared_sf.data().get(), 512);
  cuda::ptx::cp_async_bulk_commit_group();
}
```

## 性能声明

PR 正文仅给出:"当数据量足够时,本内核可充分利用 B200 的 HBM 带宽(up to 85%+)"(以带宽占用截图佐证)。未给出与其他实现的具体吞吐对比、模型或 workload 条件;正文 "Benchmarking and Profiling" 一节为空,PR 未注明其余测试条件。

## 验证与回归风险

新增 `sgl-kernel/tests/test_es_mxfp8_blockscaled_moe.py` 精度/单元测试(正文以截图形式给出结果,未含数值)。约束与风险面:仅在 `CUTLASS_ARCH_MMA_SM100_SUPPORTED`(SM100/B200)下可用,其余设备 TORCH_CHECK 报错;k、n 需 128 对齐,专家数上限 1024(受单 block 线程数限制),输出仅 BF16/FP16。CMake 与 `sgl_kernel` Python 绑定(`expert_specialization.py`)同步扩展。属特性引入,不改变既有路径行为。

## 关联

标题标注 [1/N],为 Blackwell MXFP8 系列的第一篇;同目录已有 FP8 blockwise grouped GEMM(`es_fp8_blockwise_scaled_grouped_mm`),本 PR 是其在 MXFP8 上的对应扩展。后续编号 PR 未在素材中提及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
