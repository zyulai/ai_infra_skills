# PR #21314: CUTLASS NVFP4 GEMM improvement of SM120

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-04-01 |
| 分类 | Quantization (tier: major) |
| Label | documentation, quant, sgl-kernel, blackwell, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(7 个文件未收入,详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/21314 |

## 一句话结论

用 CUTLASS profiler 为 SM120 重调 NVFP4 GEMM tile/调度启发式,并把 SM100/SM120 kernel 拆分独立调优,代表性 shape M=16, N=6144, K=5120 提速 1.197x(约 20%)。

## 动机

SM100 与 SM120 的 NVFP4 GEMM 此前共用同一份 CUTLASS 配置文件,但 SM120 不支持 multicast、部分 tile size 等特性,导致其启发式并非最优。作者用 CUTLASS profiler 对 SM120 家族做了穷举调优(tile size、cooperative vs pingpong、StreamK——其中 StreamK 对最优配置基本无影响),据此更新 SM120 的启发式选择,并为后续架构更多使用 FP4 做铺垫。

## 优化点

机制上分两步:一是代码重构,把原先单文件(约 600 行)的 `nvfp4_scaled_mm_kernels.cuh` 拆成 common(SMACheck/workspace/getSMVersion 等公共设施)、sm100(原有 KernelConfigM128/M256/Default/Fp32 及 dispatch)和 sm120(SM120 专属 config)三个头文件,使两条架构线可以独立演进和调优;二是 SM120 侧按 M 尺寸重新选择 tile 形状与调度(`sm120_fp4_config_M256` / `sm120_fp4_config_default`,cooperative BlockScaled 主循环),dispatch 以 `next_pow_2(m)` 分段。同时把 jit_kernel 的 NVFP4 benchmark 从仅 SM100 放开到 SM100/SM120,并更新量化文档中的 backend 表(`cutlass` 后端在 SM100/120 均可选)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm120.cuh: 新增,SM120 专属 GEMM 模板与按 M 分段的配置
python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm100.cuh: 新增,原 SM100 KernelConfig/dispatch 迁入
python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_common.cuh: 新增,公共宏/辅助函数(alloc_workspace_tensor、getSMVersion 等)
python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh: 瘦身为 include 转发,入口按输出 dtype dispatch 到 Sm100/Sm120 实现
python/sglang/jit_kernel/benchmark/bench_nvfp4_scaled_mm.py: _NVFP4_SUPPORTED 放开到 is_sm120_supported()
```

```cpp
struct sm120_fp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};
struct sm120_fp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};
```

## 性能声明

- 最优 shape 提速:**M=16, N=6144, K=5120 → 1.197x(约 20% speedup)**;作者称调优后 CUTLASS "almost beat cuDNN"。CUTLASS 与 cuDNN 在 128x128x256 tile 策略上对齐对比;调参范围仅覆盖 M <= 128(CUTLASS profiler 耗时过长,更大 M 留作后续)。
- 精度:以 `nvidia/Qwen3-32B-NVFP4`(modelopt_fp4)启动服务,跑 gsm8k 200 样本 x5 repeat:本 PR CUTLASS 后端 mean **0.969**,cuDNN 后端 mean **0.976**(作者附分数列表,差异在 0.007)。
- PR 附 profiler 截图与服务级 profile 命令(bench_one_batch_server, batch-size 16, input-len/output-len 1024, 模型 baseten-admin/glm-4.7-fp8-attn-fp4-mlp),但未给出端到端 tokens/s 数字。

注:性能数字来自 PR 正文截图与文字,未独立核实。

## 验证与回归风险

PR 引入/更新了 `bench_nvfp4_scaled_mm.py`(注册到 stage-b-kernel-benchmark-1-gpu-large CI 套件)覆盖 SM100+SM120;作者给出 gsm8k 精度评估流程与结果。风险点:SM120 dispatch 改为按 `next_pow_2(m)` 选配置,输出 dtype 分支(half/bf16/fp32)路径均被重定向,SM100 行为理论上不变(代码原样迁入)但依赖编译宏 `CUTLASS_ARCH_MMA_SM100_SUPPORTED` 的划分;文档同步更新了 backend 选择说明,`auto` 在 SM120 仍默认 flashinfer_cudnn,本次优化主要惠及显式选择/回退到 SGLang CUTLASS 后端的场景。

## 关联

同区域:sgl-kernel/jit_kernel NVFP4 GEMM 系列优化(Quantization 分类、blackwell 标签);pack 素材未列出具体前后序 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
