# PR #19566: [Perf] Further tunings for SM100 FP8 CUTLASS kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-06-15 |
| 分类 | Quantization |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,仅 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/19566 |

## 一句话结论

为 SM100(B200)FP8 CUTLASS GEMM 细化 M≥128 的 tile/cluster 配置并新增 M256 档,GEMM 最高 1.4x。

## 动机

#18778 引入 SM100 FP8 CUTLASS 路径时,dispatch 只有 M64/M128/default(>128)三档:128 以上全部共用 `TileShape 256x128x64`,128 档用 `128x128x64`、cluster 2x2x1。对 Llama-8B 这类形状,中大批量(M=128–4096)的 tile 纵深与 cluster 尺寸都不是最优,存在纯配置层面的提升空间。

## 优化点

纯 dispatch 配置调优,无内核逻辑变化:`sm100_fp8_config_default`(M>256)的 TileShape 从 `256x128x64` 加深到 `256x128x128`;新增 `sm100_fp8_config_M256` 档(M ∈ (128,256]),TileShape `128x128x128`、ClusterShape `2x2x1`;`sm100_fp8_config_M128` 的 TileShape 从 `128x128x64` 改为 `128x128x256`,ClusterShape 从 `2x2x1` 扩到 `2x4x1`(更大 cluster 提升 TMA 多播/L2 复用)。`cutlass_gemm_sm100_fp8_dispatch` 相应插入 `mp2 <= 256` 分支,按 2 的幂上取的 M 选 64/128/256/default 四档内核。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh: M128/M256/default 档 TileShape+ClusterShape 调整
```

```cpp
// M128 档
- using TileShape = Shape<_128, _128, _64>;
- using ClusterShape = Shape<_2, _2, _1>;
+ using TileShape = Shape<_128, _128, _256>;
+ using ClusterShape = Shape<_2, _4, _1>;
// 新增 M256 档 (128, 256]:TileShape 128x128x128,Cluster 2x2x1
```

## 性能声明

(B200;`benchmarks/kernels/bench_fp8_gemm.py --model meta-llama/Llama-3.1-8B-Instruct --tp-sizes 1`,即 #17126 的内核基准;对比列为 noquant 的 fp8-channel-w-token-a 与 fp8-tensor-w-tensor-a 两种 scheme,单位 TFLOP/s)

- 作者总结:M=128 提速 1.13–1.32x,M=256 最高 1.4x,M≥512 最高 1.2x。
- 最佳点(N=4096 K=14336):M=128 为 933.90→1235.35(1.32x)/ 931.74→1226.11(1.32x);M=256 为 1456.98→2058.74(1.41x)/ 1455.69→1914.42(1.32x)。
- 大 M 亦有收益:N=6144 K=4096 M=4096 1.45x(channel)、M=8192 1.20x(tensor);N=28672 K=4096 M=16384 1.24x(channel)。
- 存在个别回退点(如 N=6144 K=4096 M=256 为 0.94x/0.93x,N=4096 K=4096 M=8192 为 0.97x、M=2048 tensor 0.96x),总体多数配置为正收益。

## 验证与回归风险

验证依赖 `bench_fp8_gemm.py` 内核级基准(Pack 未列单测;tile/cluster 配置不改变数值语义,风险是部分形状的性能回退,上表已含个别回退点)。影响面:SM100(B200)上所有走 CUTLASS FP8 scaled_mm 的 W8A8 模型(compressed-tensors/Fp8 量化),M≤64 与 M>256 的 default 档形状也受 default tile 加深影响;其他 GPU 架构不经过该 dispatch。

## 关联

#18778(本 PR 的直接前序,引入 SM100 FP8 CUTLASS 路径);#17126(所用基准工具 bench_fp8_gemm.py);同区域:#20396(SM90 CUTLASS FP8 GEMM 扩展)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
