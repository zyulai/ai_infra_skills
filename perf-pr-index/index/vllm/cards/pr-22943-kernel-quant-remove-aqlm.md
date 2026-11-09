# PR #22943: [Kernel/Quant] Remove AQLM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-08-16 |
| 分类 | Quantization(tier: major) |
| Label | documentation, performance, ready, ci/build |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 可用但另有 10 个文件因超长未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/22943 |

## 一句话结论

移除使用率极低的 AQLM 极低比特量化方法(含 CUDA GEMM/dequant kernel 与全部绑定),以降低代码复杂度;非提速改动。

## 动机

AQLM(Additive Quantization of Language Models,自 Vahe1994/AQLM 适配、Neural Magic 修改)曾是 vLLM 支持的 2-bit 级加性码本量化方案。维护者的原话是:"We loved AQLM, but it did not see much usage so I think we should remove to reduce complexity." 即该量化路线实际用户稀少,继续维护一套专用 CUDA kernel、op binding、测试与文档的成本不值得,属于代码库健康度清理。

## 优化点

纯删除,无新增机制。删除的运行时路径包括:`csrc/quantization/aqlm/gemm_kernels.cu` 中的四个 CUDA kernel——`Code1x16MatVec`/`Code2x8MatVec`(码本查找 + `__hfma2` 的 GEMV,用 `ld.cg.global.v4.u32` 绕过 L1 以获取 >2x 的加载加速)、`Code1x16Dequant`/`Code2x8Dequant`(码本反量化),以及 `aqlm_gemm`/`aqlm_dequant` 两个 host 入口(按 nbooks×entries 分派 1x16 或 2x8 配置);`csrc/ops.h`/`csrc/torch_bindings.cpp` 中对应 op 声明与 TORCH_LIBRARY 绑定;kernel 微基准 `benchmarks/kernels/benchmark_aqlm.py`;`tests/kernels/quantization/test_aqlm.py`(opcheck)与 `tests/models/quantization/test_aqlm.py`(Llama-2-7b-AQLM-2Bit-1x16-hf 的 ground-truth 生成对比测试);`docs/features/quantization/supported_hardware.md` 中的 AQLM 行。pack 注明另有 10 个文件因超长未收录(应含 Python 侧 quantization 配置/实现文件)。对非 AQLM 用户无任何运行时行为变化;主要"收益"是编译产物更小、量化代码面收窄。

## 关键改动

```text
csrc/quantization/aqlm/gemm_kernels.cu: 删除全部 AQLM CUDA kernel(597 行)
csrc/torch_bindings.cpp: 删除 aqlm_gemm/aqlm_dequant op 注册
csrc/ops.h: 删除函数声明
benchmarks/kernels/benchmark_aqlm.py: 删除 kernel 微基准
tests/kernels/quantization/test_aqlm.py、tests/models/quantization/test_aqlm.py: 删除测试
docs/features/quantization/supported_hardware.md: 移除 AQLM 硬件支持行
```

被删 kernel 的核心机制(码本查表 GEMV,节选):

```cuda
const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
for (int i = 0; i < 8; i++) {
  uint32_t dec[4];
  // We bypass the L1 cache to avoid massive amounts of memory streaming
  // that doesn't actually help us; this brings > 2x speedup.
  asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
               : "l"((void*)&codebook[enc[i]]));
  ...
}
```

## 性能声明

PR 正文无任何 benchmark 数字(regex 检测为 none),仅有一句动机陈述。本 PR 不声明、也不影响任何路径的性能;被删代码中注释提及的 ">2x speedup" 是当年 kernel 内部的加载优化说明,非本 PR 的性能声明。PR 未注明测试条件。

## 验证与回归风险

验证方式是删除而非新增:随 AQLM 一起移除的 opcheck 测试与模型精度测试不再需要跑,CUDA/ROCm(#ifndef USE_ROCM 分支)构建不再编译 aqlm kernel。回归风险:对仍持有 AQLM 权重(如 ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf)的用户是破坏性变更——加载即报不支持;transformers 侧 AQLM 推理不受影响。对其他量化方法(AWQ/GPTQ/FP8 等)无影响,`ops.h` 中相邻声明未动。

## 关联

无。本卡批次内无同分类(Quantization)其他 PR;AQLM 最初引入与优化的 PR 编号未在素材中出现,不予虚构。
