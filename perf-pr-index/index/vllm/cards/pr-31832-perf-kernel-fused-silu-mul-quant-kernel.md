# PR #31832: [Perf][Kernel] Fused SiLU+Mul+Quant kernel for NVFP4 cutlass_moe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-01-09 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 8 个文件(fp4 量化 kernel、ops 绑定、cutlass_moe 与工具等) |
| 链接 | https://github.com/vllm-project/vllm/pull/31832 |

## 一句话结论

把 silu_and_mul 融进 NVFP4 MoE 专家量化,Qwen3-30B-A3B-NVFP4 延迟约 -2%、吞吐约 +4%。

## 动机

PR 正文 Purpose:dense NVFP4 路径早已把 silu_and_mul 融合进 NVFP4 量化(#23671),但 MoE 的专家量化尚未这样做,导致 SiLU+Mul 与量化分两次执行。本 PR 泛化相关 utils,把同一能力开放给 expert quantization,正文称可改善 latency 约 2%、throughput 约 4%。

## 优化点

在 NVFP4 专家量化 kernel `cvt_fp16_to_fp4` 上新增编译期模板参数 `FUSE_SILU_MUL`:开启时输入按 gate||up 拼接布局(宽度加倍,`inColsPerRow = colsPerRow * 2`),线程读取 gate 与 up 两个 PackedVec 后调用 `compute_silu_mul` 先算 SiLU(gate)*up,再做 FP4 量化,从而省去独立的 silu_and_mul kernel 与中间张量。为此把 `compute_silu_mul`/`silu` 等从 `activation_nvfp4_quant_fusion_kernels.cu` 泛化到共用 utils,并在 `csrc/ops.h`、`torch_bindings.cpp`、`vllm/_custom_ops.py`、`vllm/model_executor/layers/fused_moe/cutlass_moe.py` 等处接通新的融合入口与开关。

## 关键改动

```text
csrc/quantization/fp4/nvfp4_experts_quant.cu: cvt_fp16_to_fp4 增加 FUSE_SILU_MUL 模板参数,gate||up 布局下先算 SiLU*up 再量化
csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu: 泛化 silu_and_mul 相关 utils 供专家量化复用
csrc/quantization/fp4/nvfp4_quant_entry.cu / nvfp4_utils.cuh: 入口与工具适配
vllm/model_executor/layers/fused_moe/cutlass_moe.py: cutlass_moe 侧接通融合量化路径
csrc/ops.h、csrc/torch_bindings.cpp、vllm/_custom_ops.py: ops 声明与 Python 绑定
```

关键机制代码(模板参数与融合分支):

```cpp
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
...
    int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;
...
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    PackedVec quant_input;
    if constexpr (FUSE_SILU_MUL) {
      PackedVec in_vec_up =
          reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      quant_input = compute_silu_mul(in_vec, in_vec_up);
    } else {
      quant_input = in_vec;
    }
```

## 性能声明

PR 正文 Test Result(`vllm serve nvidia/Qwen3-30B-A3B-NVFP4`):延迟 benchmark `vllm bench serve --input-len 100 --output-len 100 --num-prompts 8`,MAIN → PR 的 total token throughput 2011.41 → 2050.42 tok/s、mean TTFT 33.40 → 31.20 ms、mean TPOT 7.62 → 7.53 ms;吞吐 benchmark `--num-prompts 512`,output token throughput 19335.14 → 20044.52 tok/s、total 38670.29 → 40089.04 tok/s、mean TTFT 812.20 → 776.44 ms、mean TPOT 17.91 → 17.20 ms。正文概述 "improve latency by ~2% and throughput by ~4%"。GSM8K eval(`tests/evals/gsm8k/gsm8k_eval.py`):accuracy 0.886 → 0.883、invalid responses 0.001 → 0.000。PR 未注明硬件型号与精度参数。

## 验证与回归风险

Test Plan 写 "Benchmarks and evals",正文给出上述 latency/throughput benchmark 与 GSM8K eval 前后对比,未列出新增单元测试。影响路径:NVFP4 cutlass_moe 的专家激活量化路径(Qwen3-30B-A3B-NVFP4 等 MoE 模型);改动把 SiLU+Mul 移入量化 kernel 并要求 gate||up 拼接输入,需关注未开启 FUSE_SILU_MUL 的原有路径(默认 false)保持行为不变,以及 GSM8K 0.886 → 0.883 的小幅精度变化。

## 关联

前序 PR #23671(正文所引,dense NVFP4 已实现的 silu_and_mul 与量化融合,本 PR 把它推广到 MoE 专家量化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
