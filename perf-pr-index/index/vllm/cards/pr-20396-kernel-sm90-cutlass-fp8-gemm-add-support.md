# PR #20396: [Kernel] SM90 CUTLASS FP8 GEMM: add support for swap AB + kernel tuning

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LyrisZhong |
| 合并 | 2025-07-28 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动集中在 2 个 CUDA 源文件与 1 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/20396 |

## 一句话结论

为 SM90 CUTLASS FP8 GEMM 引入 swap AB 与调优后 kernel 配置,M<=64 小 batch 场景提速 5%-25%。

## 动机

SM90 CUTLASS FP8 GEMM 在 M<=64 的小 M 场景下存在 padding 开销:当 token 数较小(如 decode batch 或小 prefill)时,按原始 A x B 布局计算会因 tile 形状不匹配而浪费算力。SM100 FP8 blockwise kernel 此前已通过 swap AB 解决同类问题(PR #18564),但 SM90 路径尚未支持。此外原有 kernel dispatch 只按 M 分档,配置粒度粗,未按 N 维度细分调优。

## 优化点

核心改动有两层。其一,新增 `cutlass_3x_gemm_sm90_fp8` 模板,带编译期 `swap_ab` flag:当启用时把 GEMM 的 A/B 两个操作数交换(问题形状从 `(m, n, k)` 变为 `(n, m, k)`),layout 同步转置,使得小 M 情况下 M 维度不再需要 padding 到 tile 大小。其二,重写 dispatch 分档:由原来的 M<=64 / M<=128 / default 三档,细化为 m<=16、m<=64 两档各自再按 N<=1280 / N>1280 分四种组合,每档用 CUTLASS profiler 结果重新调优的 TileShape 与 ClusterShape(如 `Shape<_64,_16,_256>` + `Shape<_1,_4,_1>`)。同时把 Epilogue 模板参数从类型改为 `EnableBias` bool,在 dispatch 时静态选择。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8_dispatch.cuh: 新增带 swap_ab 模板参数的 cutlass_3x_gemm_sm90_fp8,重写按 M/N 细分的 dispatch 配置与 cutlass_gemm_caller_sm90_fp8(swap 时交换 mainloop 指针与 prob_shape)
csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu: 入口改为按 EnableBias<true/false> 静态分发
tests/kernels/quantization/test_cutlass_scaled_mm.py: 与 baseline 对比容差从 rtol=1e-2 放宽到 rtol=5e-1
```

关键机制代码(dispatch 时交换操作数):

```cpp
auto prob_shape =
    swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);
...
typename GemmKernel::MainloopArguments mainloop_args =
    swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr,
                                                     a_stride}
            : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr,
                                                     b_stride};
```

## 性能声明

PR 正文给出:5% to 25% improvement across different M/N/K and quantization schemes,测试对象为 Mistral-7B-v0.1 与 Llama-2-7b-hf。测试工具为 `vllm/benchmarks/kernels/bench_fp8_gemm.py`,结果以前后对比截图形式呈现(正文未给出逐项数字表格)。精度验证:在 M∈{1..256}、N/K∈{1024..32768} 的组合上,cutlass_scaled_mm 与 triton_scaled_mm 输出无差异(rtol/atol=1e-1 内 assert 通过)。PR 未注明具体硬件型号(GPU 架构仅从代码推断为 SM90/Hopper)。

## 验证与回归风险

依赖现有 `tests/kernels/quantization/test_cutlass_scaled_mm.py`。需要注意该 PR 同时把该测试的 `rtol` 从 1e-2 放宽到 5e-1,意味着 swap AB 路径的数值偏差被显式接受,属于潜在精度回归点。影响路径:所有走 `cutlass_scaled_mm` 的 SM90 FP8 W8A8 推理(如 FP8 量化模型的 decode/小 batch 场景);非 SM90 硬件不受影响。

## 关联

同区域优化:SM100 FP8 blockwise swap AB(PR #18564,作者在正文中引用为先行工作);同分类(Quantization)的其他 FP8 GEMM 调优 PR 见索引库其余条目。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
