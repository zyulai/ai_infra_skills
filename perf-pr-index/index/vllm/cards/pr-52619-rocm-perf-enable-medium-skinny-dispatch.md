# PR #52619: [ROCm][Perf] Enable medium-skinny dispatch in RDNAHybridW4A16LinearKernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tangzzycc |
| 合并 | 2026-10-07 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/52619 |

## 一句话结论

把 medium-skinny decode GEMM 从 Triton 改走 HIP 内核，Qwen3-8B batch=3 吞吐 +63.43%。

## 动机

改动前 `RDNAHybridW4A16LinearKernel` 仅在 `M <= 5` 且 `K * M <= 32768` 时把 decode GEMM 路由到 HIP `wvSplitK_int4_g`。但 C++ 算子支持最多 5 个激活行（`N_in <= 5`）且接受 `K_in * N_in` 到 `int(max_lds_len * 1.2)`（`max_lds_len=32768`，即 39321 元素）；对超过常规 LDS 容量的输入，medium 内核把激活前缀留在 LDS、其余从 global memory 读。Python `M` 对应 C++ `N_in`。因此 `K*M` 落在 (32768, 39321] 的受支持形状被错误回退到 Triton。本 PR 把 Python dispatch 上限对齐 C++ 契约，不改任何 C++ 内核。

## 优化点

新增 `MEDIUM_SKINNY_LIMIT_ELEMENTS = int(LDS_CAPACITY_ELEMENTS * 1.2)`（=39321），把 `_rdna_hybrid_w4a16_apply_impl` 判定由 `K * M <= LDS_CAPACITY_ELEMENTS` 改为 `K * M <= MEDIUM_SKINNY_LIMIT_ELEMENTS`，使 medium-skinny 形状走 HIP `wvSplitK_int4_g` 而非 Triton prefill kernel。同步更新 docstring 路由说明；新增 dispatch-boundary 测试（regular / medium / above-medium / batch-too-large）与 FP16/BF16 medium 内核对 FP32 参考的数值正确性覆盖。无用户可见 API 或配置变化。

## 关键改动

```text
vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py: 新增 MEDIUM_SKINNY_LIMIT_ELEMENTS 常量，_rdna_hybrid_w4a16_apply_impl 的 HIP/Triton dispatch 阈值由 LDS_CAPACITY_ELEMENTS 放宽到该常量，并更新 docstring。
tests/kernels/quantization/test_rdna_hybrid_w4a16.py: 新增 test_rdna_hybrid_w4a16_dispatch_boundaries（4 个形状）与 test_rdna_hybrid_w4a16_medium_skinny_matches_reference（FP16/BF16、has_zp 对 FP32 参考）。
```

```text
MAX_SKINNY_BATCH_SIZE = 5
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements
MEDIUM_SKINNY_LIMIT_ELEMENTS = int(LDS_CAPACITY_ELEMENTS * 1.2)

if M <= MAX_SKINNY_BATCH_SIZE and K * M <= MEDIUM_SKINNY_LIMIT_ELEMENTS:
```

## 性能声明

测试条件：GPU AMD Radeon 8060S / gfx1151，128 GB unified memory（96 GiB VRAM + 32 GiB host-visible），Ubuntu 24.04.4 LTS，ROCm 7.2.3；`VLLM_ROCM_USE_AITER=0` + `ROCM_ATTN`，eager（`enforce_eager=True`，HIP Graph 与 `torch.compile` 关闭），prefix caching off，固定 prompts 与 random seed，512 output tokens/request、`temperature=0` + `ignore_eos=True`，1 warmup + 3 measured runs。Qwen3-8B-GPTQ-Int4（batch=3，`M=3, K=12288, N=4096`，`K*M=36864`，Before=Triton / With=HIP medium）：Aggregate decode throughput 41.70 → 68.16 tok/s（+63.43%），per-request 13.90 → 22.72 tok/s（+63.43%），batch wall time 36.92 → 22.60 s（-38.79%）。Qwen3-4B-GPTQ-Int4（batch=4，`M=4, K=9728, N=2560`，`K*M=38912`）：Aggregate 119.30 → 126.83 tok/s（+6.31%），per-request 29.83 → 31.71 tok/s（+6.31%），batch wall time 17.19 → 16.16 s（-5.99%）。Direct kernel timing：`M=3,K=12288,N=4096` Triton 0.643 ms vs HIP medium 0.388 ms（1.66x）；`M=4,K=9728,N=2560` 0.136 ms vs 0.071 ms（1.93x）。

## 验证与回归风险

Focused tests 8 passed、83 deselected；complete hybrid W4A16 test file 91 passed、14 warnings；applicable pre-commit hooks 与 `git diff --check` 通过。新增 dispatch 边界与 FP16/BF16 medium 内核数值覆盖。风险：改动仅限 gfx11/gfx12 的 hybrid 路径（`on_gfx1x()`），仅放宽 Python dispatch 阈值、未改 C++ 内核；正文声明无用户可见 API 或配置变化，无需文档更新。

## 关联

#46186、#51511、#50618（正文列为相关 PR，均未修改此 hybrid dispatch 阈值）；正文称搜索 medium-skinny、wvSplitK W4A16、RDNA hybrid W4A16 无重复 open PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
