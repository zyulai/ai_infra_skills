# PR #44075: [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JartX |
| 合并 | 2026-06-06 |
| 分类 | MoE（tier: major） |
| Label | ci/build, ready, rocm, verified |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾注明 "8 file(s) dropped from diff for size (see PR for full diff)" |
| 链接 | https://github.com/vllm-project/vllm/pull/44075 |

## 一句话结论

RDNA3(gfx1100) 上 W4A16 MoE 换用 fused HIP kernel，Qwen3-30B-A3B AWQ 在 batch 8–32 吞吐达 2.15–2.17×（单流 1.23×）。

## 动机

正文 Summary 说明：本 PR 为 RDNA3 (gfx1100) 的 W4A16 MoE 新增原生 HIP kernel，替代 Triton 的 `fused_moe_kernel_gptq_awq` 路径。该 kernel 复用 dense W4A16 kernel 已有的反量化与点积原语：`v_dot2_f32_f16` / `v_dot2_f32_bf16`、exllama bit-trick 反量化、64 位 CAS atomic 输出。正文未进一步说明原 Triton 路径存在何种具体瓶颈，benchmark 侧显示原路径在 batched decode 下吞吐明显更低。

## 优化点

`csrc/rocm/moe_q_gemm_rdna3.cu` 把 expert routing 与 W4A16 GEMM 合并为单次 kernel launch，模板化 `BLOCK_SIZE_M = {1, 2, 4, 8}`；bf16 的 `M=1` 快路径跳过 LDS staging（直接全局读，配合 `v_dot2_f32_bf16` opacity trick 绕过 InstCombine）。新增 `output_topk` 参数把 `moe_sum` 融进 w2 kernel，通过 atomics 直接写 `out[token_id / top_k]`，省掉一次独立 kernel launch 与中间 buffer。Decode 侧小 `M` 用 `BLOCK_SIZE_M = 1`（消除约 75% padding 浪费），并预分配 w1/act buffer（消除每层 `torch.zeros`）。`_try_get_rocm_moe_method()` 按优先级检查架构与 op 可用性，无原生 kernel 时回落到 Triton WNA16，便于后续加入 RDNA4/CDNA 分支；Build guard 保证 `.cu` 仅在 `VLLM_GPU_ARCHES` 含 `gfx1100` 时编译（torch op 注册在 `#ifdef VLLM_ROCM_GFX1100` 下，Python 侧以 `hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3")` 检查可用性）。

正文还说明 `csrc/rocm/moe_q_gemm_rdna3.cu` 是 639 行新文件，权重格式与 dense kernel 一致（`[E, K/8, N] uint32 shuffled`、`[E, groups, N] scales`、`[E, groups, N/8] packed zeros`），输出以 64 位 packed CAS atomic-add 直接写到预清零的输出张量、不使用 FP32 scratch buffer；每个 block 处理分给一个 expert 的 `BLOCK_SIZE_M` 个 tokens，覆盖 N 列与 K 位置的一个 tile。`THREADS_X=256`（wave32 上 8 个 wave）、`BLOCK_KN_SIZE=256`，每线程处理 4 个 N 列；bf16 `M=1` 快路径跳过 LDS staging 直接全局读。

## 关键改动

```text
csrc/rocm/moe_q_gemm_rdna3.cu: 新增 fused MoE W4A16 HIP kernel（639 行），routing+GEMM 单 kernel、64 位 CAS atomic 输出
csrc/rocm/ops.h: 声明新 op
csrc/rocm/torch_bindings.cpp: 注册 torch op（#ifdef VLLM_ROCM_GFX1100）
tests/kernels/quantization/test_rdna3_compile_guards.py: 编译 guard 相关测试
```

```cpp
// Fused MoE W4A16 GPTQ kernel for RDNA3 (gfx1100).
// Combines expert routing (sorted_token_ids / expert_ids) with the RDNA3
// W4A16 dequant+dot from q_gemm_rdna3.cu into a single kernel launch.
// Design: THREADS_X=256 (8 waves on wave32), BLOCK_KN_SIZE=256, each thread
// handles 4 N columns. Output via 64-bit packed CAS atomic-add directly to
// the pre-zeroed output tensor (no FP32 scratch buffer).
```

## 性能声明

测试条件（Benchmarks 段）：1× RX 7900 XTX (gfx1100)，EvalScope `perf`，`openqa` dataset，50 requests；模型 `cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`（MoE，128 experts / 8 active，W4A16 AWQ）；对比 Native HIP kernel 与 upstream Triton `fused_moe_kernel_gptq_awq`。Throughput (completion tok/s)：`max_num_seqs=1` Triton 82.84 / HIP 102.01 / **1.23×**；`=8` 186.99 / 401.94 / **2.15×**；`=32` 348.49 / 755.62 / **2.17×**。Per-request latency (max_num_seqs=32)：TPOT 65.4→31.9 ms (2.05×)、Latency avg 65.3→33.4 s (1.95×)、TTFT avg 14,888→8,634 ms (1.72×)。正文结论："The speedup scales with concurrency: ~1.2× single-stream, growing to ~2.2× under batch (8–32 seqs)"。

## 验证与回归风险

PR 依赖/引入的测试：EvalScope `perf`（openqa，50 requests）吞吐与延迟对比；ACC EVAL 用 lm_eval 跑 gsm8k（`vllm serve ... --attention-backend TRITON_ATTN --gpu-memory-utilization 0.90 --max-model-len 16384 --tensor-parallel-size 1`，`local-chat-completions`，`num_concurrent=35`），对比 exact_match：HIP flexible-extract 0.868±0.0215 / strict-match 0.840±0.0232，TRITON 0.848±0.0228 / 0.836±0.0235；Testing 段以 `evalscope eval --datasets openqa` 检查 output parity；新增编译 guard 测试。可能影响路径：仅 gfx1100 / `VLLM_GPU_ARCHES` 含 gfx1100 时编译并命中新 op，其他架构回落到 Triton WNA16；PR 未注明其他回归风险。

## 关联

无（PR 正文与 diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
