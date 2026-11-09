# PR #43534: [CPU][Perf] Enable fused kernels for GDN's gated delta rules

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2026-06-02 |
| 分类 | Attention / Kernel |
| Label | ready, ci/build, cpu |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/43534 |

## 一句话结论

GDN gated delta rule 融合内核改为 ISA 无关,非 AMX CPU 走 OpenBLAS/PyTorch BLAS fallback,Neoverse-V2 吞吐 +50%。

## 动机

此前从 sglang-kernels 移植的 GDN(gated delta net)chunk gated delta rule 内核把 GEMM 硬绑定在 Intel AMX 的 `brgemm`/VNNI pack 路径上,`can_use_brgemm` 无条件返回 true,导致无 AMX 的 CPU(如 Arm Neoverse)要么走错路径要么无法用上融合内核;同时 Python 侧还有一条独立的 `cpu_gdn_attention_core_amx` 分支,AMX 与非 AMX 行为分裂。另有一个 sigmoid gating 内核读错 beta 指针的正确性 bug。

## 优化点

将 brgemm 调用全部包进 `if constexpr (brgemm_supported())`(`__AVX512F__ && __AVX512BF16__ && __AMX_BF16__` 才为 true),否则调用新头文件 `blas_gemm.h`:PyTorch wheel 内含 OpenBLAS 时用 `sbgemm_`/`sgemm_` 做 bf16:bf16→fp32 GEMM,否则退回 PyTorch 的 `cpublas::gemm_no_downcast_stub` 参考路径;`can_use_brgemm` 各 dtype 特化都加上 `brgemm_supported()` 前置条件。Python 侧删除 AMX 专用入口 `cpu_gdn_attention_core_amx`,统一走单一 fused kernel 路径,并断言 BF16。同时修复 `fused_sigmoid_gating_delta_rule_update_kernel_impl` 中 `b_ptr[ni]` 的错误索引(应为 `b_ptr[bi * v_num_heads + ni]`,对齐 sglang#26634)。

## 关键改动

```text
csrc/cpu/sgl-kernels/blas_gemm.h: 新增 OpenBLAS/PyTorch BLAS fallback GEMM 封装
csrc/cpu/sgl-kernels/fla.cpp: chunk_gated_delta_rule 各 GEMM 点按 brgemm_supported() 分支
csrc/cpu/sgl-kernels/gemm.h: brgemm_supported() + can_use_brgemm 加 ISA 条件
vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py: 删除 AMX 专用路径,统一 fused kernel
tests/kernels/mamba/cpu/test_cpu_gdn_ops.py: 新增三个 fused op 的数值对比测试
```

## 性能声明

Neoverse-V2 上融合内核带来 50% higher throughput(vllm bench throughput:128 prompts,random 数据集,input 256 / output 128,max-model-len 4096,max-num-batched-tokens 4096,模型 Qwen/Qwen3.5-9B,TP=1,关闭 prefix caching,`--load-format dummy`,5 次预热)。

## 验证与回归风险

新增 `tests/kernels/mamba/cpu/test_cpu_gdn_ops.py`:对 `fused_gdn_gating_cpu`、decode 路径 `fused_sigmoid_gating_delta_rule_update_cpu`、prefill 路径 `chunk_gated_delta_rule_cpu` 做参考实现数值对比(多种 batch/seq_len/head 组合,atol/rtol 1e-2 或 1e-4)。风险集中在 CPU 平台:非 x86/无 OpenBLAS 环境走新的 fallback 路径,GDN 现强制 BF16;x86 AMX 路径逻辑不变但入口统一,speculative decode 场景仍不支持(CPU GDN attention 有断言)。causal conv 的加速留待后续 PR。

## 关联

同区域:CPU GDN/causal conv 内核线(后续 PR 将加速 causal conv);修复依据 sgl-project/sglang#26634。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
