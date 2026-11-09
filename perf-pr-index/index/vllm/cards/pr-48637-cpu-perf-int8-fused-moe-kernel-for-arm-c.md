# PR #48637: [CPU][Perf] INT8 Fused MoE Kernel for Arm CPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2026-07-27 |
| 分类 | MoE (tier: major) |
| Label | ready, ci/build, cpu |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 部分可用——因篇幅丢弃了 10 个文件,核心为新增 647 行的 cpu_fused_moe_int8.cpp |
| 链接 | https://github.com/vllm-project/vllm/pull/48637 |

## 一句话结论

为 Arm CPU 新增 W8A8 INT8 fused MoE kernel(AdvSIMD smmla 微内核),比现有 BF16 kernel 快约 1.9x,使量化 MoE checkpoint 可跑在 Arm CPU 上。

## 动机

Arm CPU 路径此前只有 BF16 fused MoE kernel,无法运行 W8A8(per-token input、channelwise weights)的量化 MoE checkpoint,例如 `RedHatAI/Qwen3-30B-A3B-Thinking-2507-quantized.w8a8`。作者尝试直接在既有 BF16 grouped GEMM kernel 上扩展 INT8 支持但代码难以维护,因此选择复用其结构另写一个 INT8 kernel。

## 优化点

新增 `fused_moe_int8_impl`:沿用 BF16 fused MoE 的两级结构(w13 GEMM + 门控激活、in-place 逐 token 重新量化、w2 GEMM),但 GEMM 主体换成 INT8 微内核 `MicroGemmINT8`,基于 AdvSIMD `smmla` 指令(int8 dot-product 矩阵乘),仅在编译期检测到 `ARM_I8MM_SUPPORT` 与 `ARM_BF16_SUPPORT` 时启用。工程细节:每 token 先 `quantize_row` 量化到 int8 并记录 scale,w13 输出经 `dequantize_tile` 反量化回 FP32/BF16 后应用门控激活,再 in-place 量化作为 w2 输入;`w13`/`w2` 的 n-tile 大小按可用 L2 cache 容量与线程数动态计算,配合 `ScratchPadManager` 统一 scratch buffer;线程间用 Counter 做动态任务分发(专家 x 输出分块)。同时把 `FusedMOEAct` 与 silu/gelu/swigluoai 激活实现从 `cpu_fused_moe.cpp` 抽到共享头文件 `cpu_fused_moe_activations.hpp` 供 BF16/INT8 两套 kernel 复用。

## 关键改动

```text
csrc/cpu/cpu_fused_moe_int8.cpp: 新增 INT8 fused MoE kernel(quantize -> smmla GEMM -> dequant + act -> requant -> w2 GEMM)
csrc/cpu/micro_gemm/cpu_micro_gemm_int8_neon.hpp: MicroGemmINT8 微内核(pack_weight/quantize_row/dequantize_tile,gemm 提供),diff 中被丢弃
csrc/cpu/cpu_fused_moe_activations.hpp: 新增,从 cpu_fused_moe.cpp 抽出的共享门控激活实现
csrc/cpu/cpu_fused_moe.cpp: 改用共享激活头文件
```

ISA 分派与量化循环(节选):

```cpp
#if defined(ARM_I8MM_SUPPORT) && defined(ARM_BF16_SUPPORT)
  #include "cpu/micro_gemm/cpu_micro_gemm_int8_neon.hpp"
  // NEON_DISPATCH -> MicroGemmINT8<cpu_utils::ISA::NEON, SCALAR_TYPE>
...
#pragma omp parallel for
  for (int32_t token_idx = 0; token_idx < token_num; ++token_idx) {
    gemm_t::quantize_row(input + token_idx * input_size_13,
                         input_quant_buffer + token_idx * input_size_13,
                         input_scale_buffer[token_idx], input_size_13);
  }
```

## 性能声明

PR 正文声明:该 INT8 fused MoE kernel 对 gpt-oss、qwen3-30b-a3b、gemma4-26b-a4b 比现有 BF16 kernel 快约 ~1.9x。测试条件:PR 未注明硬件型号、线程数、batch/token 规模与测量方式(kernel 级或 e2e)。

## 验证与回归风险

Test Plan 仅写 "CI",正文未给出测试命令与结果。回归风险:新 kernel 由 `ARM_I8MM_SUPPORT && ARM_BF16_SUPPORT` 编译期开关与 NEON ISA 分派保护,不满足条件的 ISA 会落入 `TORCH_CHECK(false, "Invalid CPU ISA type.")`;scratch buffer 布局按 L2 大小与线程数动态推导,是性能与正确性的敏感点;INT8 两级 GEMM 的中间 requant 会带来与 BF16 路径不同的数值特性(W8A8 量化模型本身语义)。激活代码抽到共享头文件属纯重构,`diffusiongemma` 相关的 tanh 实现注释保留。

## 关联

同区域 Arm CPU 优化:PR #37469(BF16 GELU LUT)、PR #27244(cpu fused moe perf)。PR 正文未引用其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
