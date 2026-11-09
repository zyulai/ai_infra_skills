# PR #20887: CUTLASS FP8 Blockwise GEMM improvement of SM120

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-03-22 |
| 分类 | Quantization (tier: major) |
| Label | sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用(单文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/20887 |

## 一句话结论

SM120 FP8 blockwise GEMM 对 M≤64 启用 CUTLASS pingpong schedule,小 M 下较 cooperative 约快 2x。

## 动机

SM120 的 fp8 blockwise GEMM kernel 此前使用 `KernelScheduleAuto`,而该策略在 SM120 上总是选中 cooperative kernel。对小 M(解码类)shape 这不是最优解:pingpong schedule 约快 2x,存在明确的性能机会。作者参考 CUTLASS 官方示例实现。

## 优化点

在 sgl-kernel 的 `fp8_blockwise_gemm_kernel.cu` 中做双路径分发:新增编译期条件 `kCanUsePingpong`(要求 `64 % ScaleGranularityM == 0`)与运行时 `m <= 64` 判断,M≤64 走 `KernelTmaWarpSpecializedBlockwisePingpongSm120`(64x128x128 tile)的 pingpong 路径,M>64 走 cooperative 路径。cooperative 的 schedule 由 `KernelScheduleAuto` 显式改为 `KernelScheduleSm120Blockwise`,以规避 refcheck 输出相对误差爆炸的问题(作者疑似 CUTLASS 库问题,原因未明)。pingpong 执行失败时回退 cooperative。同时把 kernel 配置重构为 `run_gemm` lambda,两种 schedule 共享同一套参数构造与启动逻辑。

## 关键改动

```text
sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu: launch_sm120_fp8_blockwise_scaled_mm 新增 pingpong/cooperative 双路径与 m<=64 分发,重构 setup
```

```cpp
if (m <= 64) {
  status = run_gemm(PingpongGemmKernel{});
  if (status != cutlass::Status::kSuccess) {
    status = run_gemm(CooperativeGemmKernel{});
  }
} else {
  status = run_gemm(CooperativeGemmKernel{});
}
```

## 性能声明

RTX 5090 (SM120),Qwen/Qwen3.5-27B-FP8 shapes(M=1..512),N=1536、K=5120,kernel 延迟对比:M=8 本 PR 0.034 ms、FlashInfer 0.063 ms、Triton 0.041 ms;M=64 为 0.034/0.063/0.041 ms;M=128 为 0.063/0.063/0.042 ms;M=512 为 0.063/0.063/0.043 ms(本 PR/FlashInfer/Triton)。

- BS=1 decode 速度:before(cooperative)34.14 token/s(29.996 s,1024 tokens);after(pingpong)52.11 token/s(19.652 s,1024 tokens)。
- gsm8k 200 examples、5 repeat:before mean_score 0.987,after 0.99。

## 验证与回归风险

Checklist 注明已加 unit tests 并提供精度/速度基准;正文给出 gsm8k `run_eval` 命令与前后精度。风险面:SM120 上所有 fp8 blockwise GEMM 调用受影响——M>64 的 cooperative schedule 从 Auto 显式改为 `KernelScheduleSm120Blockwise`(为绕开 refcheck 误差问题,作者未查明根因),M≤64 改走 pingpong;pingpong 仅在 `64 % ScaleGranularityM == 0` 时编译启用,失败时回退 cooperative 保证功能兜底。

## 关联

正文未提及其他关联 PR 编号;作者建议后续可更改 SM120 默认 GEMM backend(未附 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
