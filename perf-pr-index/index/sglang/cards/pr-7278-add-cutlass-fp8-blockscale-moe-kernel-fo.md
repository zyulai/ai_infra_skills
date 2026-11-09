# PR #7278: Add CUTLASS FP8 Blockscale MoE kernel for Hopper architecture

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ayrnb |
| 合并 | 2025-07-03 |
| 分类 | MoE |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/7278 |

## 一句话结论

为 Hopper(sm90)新增 CUTLASS FP8 Blockscale 分组 GEMM MoE 内核,常见配置下比 DeepGEMM 快约 5%-6%。

## 动机

此前 sgl-kernel 的 `fp8_blockwise_scaled_grouped_mm` 只支持 sm100(Blackwell),Hopper 上的 FP8 blockwise 量化 MoE 只能走 DeepGEMM。作者用 PR 附带的 benchmark 对比后得出:CUTLASS FP8 Blockscale MoE 在部分常见配置下比 DeepGEMM 快约 5%-6%,值得在 Hopper 落地;同时明确 CUTLASS 在 Small K 场景不如 DeepGEMM,应在分派时跳过。

## 优化点

在 `fp8_blockwise_moe_kernel.cu` 中新增 sm90 路径:`launch_sm90_fp8_blockwise_scaled_group_mm` 用 CUTLASS CollectiveBuilder 组装 FP8(e4m3)输入、blockwise scale(Sm90BlockwiseScaleConfig<1,128,128>)、ptr-array 分组 GEMM 内核;调度选 `KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum` + `PtrArrayTmaWarpSpecializedPingpong` epilogue——Pingpong 不需要 mainloop 结束后同步 warp group,相比 Cooperative 异步持续占用 TensorCore。MmaTileShape 64x128x128、ClusterShape 2x1x1。分派条件 `sm_version == 90 && a.size(1) > 256`:Small K(≤256)场景 CUTLASS 用低效 TensorCore 指令(64x128x32 vs DeepGEMM 64x160x32)且 epilogue 有多余 FFMA,故显式跳过、回退其他实现,并报错信息中带上 K 大小。另附 DeepSeek-R1/Qwen3-235B 各 prefill/decode 形状下的 deepgemm vs cutlass 对比脚本。

## 关键改动

```text
sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu: 新增 sm90 launcher 与分派(K>256 才走 CUTLASS)
sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py: 新增 deepgemm vs cutlass 分形状 benchmark
sgl-kernel/tests/test_fp8_blockwise_moe.py: 测试扩展到 sm90(CUDA>=12.8),atol 5e-4→1e-3
```

关键片段(分派与调度配置):

```cpp
if (sm_version == 90 && a.size(1) > 256) { ... }
using MmaTileShape = Shape<_64, _128, _128>;
using ClusterShape = Shape<_2, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
```

## 性能声明

PR 正文:CUTLASS FP8 Blockscale MoE kernel 在部分常见配置下比 DeepGEMM 快约 5%-6%(依据 `python3 sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py` 结果截图,具体逐形状数字在图片中,文本未列出)。结论性分析:EP 与部分 TP+prefill 配置下 CUTLASS 相对 DeepGEMM 有增益;Small K 场景 DeepGEMM 更优,已在代码中跳过。PR 未注明基准硬件型号;benchmark 脚本覆盖 DeepSeek-R1(TP8/TP16/EP8/EP16,chunk 4096-16384,bs 32-256)与 Qwen3-235B-A22B-FP8(TP4)的 gateup/down 形状,共 14 组。

## 验证与回归风险

`test_fp8_blockwise_moe.py` 的 skip 条件从仅 sm100 扩展为 sm100 或 sm90(sm90 需 CUDA >= 12.8),数值容差 atol 从 5e-4 放宽到 1e-3(sm90 内核与 sm100 数值特性不同)。风险面:影响所有走 `fp8_blockwise_scaled_grouped_mm` 的 FP8 blockwise 量化 MoE 模型(DeepSeek-R1、Qwen3-235B 等);K≤256 显式拒绝并抛错(报错信息含 K),依赖上层分派选择其他实现;同时删除了 `problem_sizes` 的部分形状/dtype 校验,传入非法 problem_sizes 时不再提前报错。Future work(正文):EP 模式支持、去掉 epilogue 多余 LinearCombination。

## 关联

同分类(MoE):PR #42654(NVFP4 routed TRT-LLM runner deferred finalize)、PR #32340(DSV4 shared experts fusion)。同量化技术栈:PR #12316(CPU FP8 GEMM 反量化加速)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
