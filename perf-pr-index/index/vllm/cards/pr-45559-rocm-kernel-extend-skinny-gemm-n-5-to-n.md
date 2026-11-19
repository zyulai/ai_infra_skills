# PR #45559: [ROCm][Kernel] Extend skinny gemm N=5 to N=8 cases on GFX12 (RDNA4) using SWMMAC optimization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @keneoneth |
| 合并 | 2026-11-16 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, ci/build |
| Diff | pack 标注 +None/-None, None files；pack 末尾说明「5 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/45559 |

## 一句话结论

在 ROCm GFX12（RDNA4）上用 SWMMAC 稀疏矩阵乘指令新增 skinny GEMM（N=5–8）专用 kernel 路径，36 个实测 serving 场景全部提速：请求吞吐平均 +10–23%、输出 token 吞吐 +7–23%、E2EL 平均降 6–19%。

## 动机

ROCm/GFX12（RDNA4）常用于本地或小规模 serving：单用户或少量用户、请求率较低，decode batch 保持小 N 而不会填满大 batch，此时 decode GEMM 落在 N=5–8 的 skinny 形状上。skinny GEMM 会因 padding 或维度未填满而浪费矩阵指令吞吐。PR 沿用 Hugging Face 博客《Creating custom kernels for the AMD MI300》的方向（MI300X 上 ≤8 行输入的 skinny GEMM 优化），并参考 IREE issue《Optimizing skinny GEMM workloads with sparse matrix instructions》（CDNA sparse MFMA）。作者强调这不是两者的直接移植，而是针对 vLLM ROCm/GFX12 路径、按实测小 N decode GEMM 形状调优的 SWMMAC 实现。

## 优化点

- **核心思路**：把每条逻辑 skinny A 行映射为两条相邻的 SWMMAC sparse 行——第一条 sparse 行从 `A[n, :]` 选一组 16-bit pairs，第二条选互补 pairs；两条 sparse 行都乘同一个正常 `B[32x16]` fragment 得到两个部分累加行，epilogue 再把相邻部分行相加恢复最终逻辑输出行。这种交错 sparse-row 布局让 N=5–8 时 SWMMAC 指令形状不被闲置。
- **两条 kernel 路径**：`swmmacGemmRegular`（每个 active wave 算一个 16 列输出 tile，B 用双 LDS panel ping-pong：预取下一 panel 到 VGPR、从 LDS 算当前、提交到备用 panel）与 `swmmacGemmSplitK`（不同 wave 算同一输出 tile 的不同 K shard，部分和经 LDS 归约；当 regular M tiling 填不满 GPU 且 K 足够长（kblocks ≥ 64）时启用）。
- **支持范围**：N=5/6/7/8，FP16 与 BF16（`__builtin_amdgcn_swmmac_f32_16x16x32_f16/bf16_w32`），多 M/K 尺寸，对齐与不对齐 M，大形状；bias 支持 scalar / per-column / per-row / full [N, M] 四种广播。N=4 也评估过但无收益，故优化路径限定 N=5–8。
- **保持回退**：不支持的形状继续走既有 fallback 路径；host launcher `swmmacGEMM` 仅在 gfx1200/gfx1201 上可用（`on_gfx120x` 检查）。

## 关键改动

```text
csrc/rocm/ops.h: + torch::Tensor swmmacGEMM(const at::Tensor& in_a, const at::Tensor& in_b,
  +                const std::optional<at::Tensor>& in_bias,
  +                const int64_t logical_M, const int64_t CuCount);
csrc/rocm/skinny_gemms.cuh:
  + inline bool on_gfx120x()  // gfx1200 / gfx1201
  + using swmmac_i16x8 = short __attribute__((ext_vector_type(8))); ... (6 个向量类型别名)
  + static constexpr int32_t kSwmmacSelect01 = static_cast<int32_t>(0x44444444u);
  + static constexpr int32_t kSwmmacSelect23 = static_cast<int32_t>(0xEEEEEEEEu);
  + struct swmmac_traits<__hip_bfloat16> { mma -> __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a, b, acc, sparse_idx); }
  + struct swmmac_traits<half>          { mma -> __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, acc, sparse_idx); }
  + load_swmmac_a_direct()   // 每条逻辑 A 行 -> 两条 SWMMAC sparse 行
  + prefetch_swmmac_b_panel() / commit_swmmac_b_panel()  // B panel: VGPR 预取 + LDS ping-pong, nontemporal load
  + __global__ swmmacGemmRegular<...>(...)   // static_assert(N >= 5 && N <= 8)
  + __global__ swmmacGemmSplitK<...>(...)    // SPLIT_K=4, partials_lds[SPLIT_K][N][columns_per_wave + 1]
csrc/rocm/skinny_gemms_swmmac.cu: 新增 host launcher（constexpr tile_columns=16, k_per_instruction=32, split_k=4,
  regular_waves_per_block=4; use_splitk = regular_is_underfilled && total_num_kblocks >= 64; K 必须 32 整除）
csrc/rocm/torch_bindings.cpp: + rocm_ops.def("swmmacGEMM(Tensor in_a, Tensor in_b, Tensor? in_bias, int logical_M, int CuCount) -> Tensor")
tests/kernels/quantization/test_rocm_skinny_gemms.py: + NKM_FACTORS_SWMMAC 36 组 (n,k,m) 形状 + test_rocm_swmmac_gemm_kernel 参数化（bias_mode 0/1/2 × bf16/fp16 × 对齐/非对齐）
```

## 性能声明

- **测试硬件**：AMD Radeon AI PRO R9700（gfx1201，GFX12/RDNA4）×1；CPU AMD Ryzen 9 9950X ×32；RAM 123Gi。模型 Qwen/Qwen3-4B。
- **基准范围**：concurrency 5/6/7/8；输入/输出形状 512/128、1024/256、2048/128、2048/512、4096/512、8192/512、1024/2048、512/4096、512/8192；baseline 为 commit 6bdabbad5bce747865fd3a249658518a4269cc22；`vllm bench serve --request-rate inf --max-concurrency C --num-warmups 16`。
- **总结**（按 workload 对 concurrency 5–8 平均）：request throughput 提升约 10–23%，output token throughput 提升约 7–23%，mean E2EL 降约 6–19%，TTFT 持平。分 workload 平均：512/128 请求吞吐 +23.0%、E2EL −19.0%；8192/512 请求吞吐 +9.8%、E2EL −6.5%；1024/2048 请求吞吐 +20.8%；512/8192 E2EL −14.5%。
- **明细示例（原文数字）**：C5 512/128：Req/s 1.83 → 2.24（+22%）、Out tok/s 233.69 → 286.76（+23%）、E2EL 2737.24 → 2229.83 ms（−19%）；C8 1024/256：Req/s 1.38 → 1.69（+22%）、E2EL 5779.56 → 4723.14 ms（−18%）；C5 512/8192：Req/s 0.02 → 0.03（+50%，Out tok/s +18%）；C6 512/8192：Req/s Δ 0%（Out tok/s +17%）。
- **GSM8K（Qwen3-4B，--max-num-seqs 8，lm_eval num_concurrent=8，5-shot）**：Baseline flexible-extract 0.8613 ± 0.0095 / strict-match 0.8650 ± 0.0094（1319 requests in 11:13, 1.96 it/s）；SWMMAC 0.8605 ± 0.0095 / 0.8666 ± 0.0094（1319 requests in 09:48, 2.24 it/s）。精度相当，SWMMAC 运行更快。

## 验证与回归风险

单测 `pytest -v tests/kernels/quantization/test_rocm_skinny_gemms.py::test_rocm_swmmac_gemm_kernel`：216 passed, 16 warnings in 52.55s，覆盖 N=5–8 × FP16/BF16 × 多 M/K × 对齐/非对齐 M × 大形状 × 3 种 bias 模式。回归风险：新路径仅在 gfx1200/gfx1201 生效（`on_gfx120x` + `#if defined(__gfx1200__) || defined(__gfx1201__)`，host 编译用匹配签名的 stub）；K 必须 32 整除、N 限 [5,8]，越界由 TORCH_CHECK 拦截；split-K 启发式（underfilled + kblocks ≥ 64）与 grid 上限（cu_count × 2 / cu_count × split_k × 2）对极端形状的行为依赖调参；TTFT 在部分大输入场景有波动（如 C8 4096/512 +2.8%、C7 4096/512 +13%）。PR 正文未给出其他已知回归。

## 关联

- 参考方向：Hugging Face 博客 https://huggingface.co/blog/mi300kernels ；IREE issue https://github.com/iree-org/iree/issues/22863 。
- baseline commit：6bdabbad5bce747865fd3a249658518a4269cc22（[CI/Build] Enable Step3p7ForConditionalGeneration testing #43956）。
- pack 未提供其他 vLLM PR/issue 关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
