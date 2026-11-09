# PR #40977: [ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgehre-amd |
| 合并 | 2026-07-14 |
| 分类 | Quantization (tier: major) |
| Label | performance, rocm, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾标注 "9 file(s) dropped from diff for size"(见 PR 完整 diff),diff 保留 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/40977 |

## 一句话结论

ROCm gfx11/12 上按 M 路由 HIP skinny decode / Triton prefill 的 W4A16 kernel,Strix Halo TPOT 最高 4.77×。

## 动机

正文说明为 ROCm gfx11/gfx12 增加 hybrid W4A16 linear kernel,按 batch size 在两种 GEMM 实现间路由,以同时覆盖单 token decode 与 prefill / batched inference 两类形态:M <= 5 走 HIP `wvSplitK_int4_g` skinny GEMM(面向单 token decode),M > 5 走 Triton fused dequant GEMM(面向 prefill / batched inference)。

## 优化点

两条路径共享同一个 AWQ-style packed shuffle 布局 `[N, K//8]` 的权重 tensor;支持对称(uint4b8、bias=8)与非对称(uint4、per-group zero-points)量化,group size 32/64/128。Triton kernel 含针对 gfx1151(Strix Halo、RDNA 3.5)与 gfx1201(RDNA 4)的 tile-size heuristic,基于 Qwen3-4B 与 Llama-3.1-8B-AWQ 的真实权重形状,并加 shape-specific overrides。HIP skinny decode kernel 对 bf16 模型用 marlin-style magic-number dequantization trick,避免 gfx1151 上昂贵的 fp32 round-trip。

## 关键改动

```text
csrc/rocm/skinny_gemms_int4.cu: HIP wvSplitK_int4_g skinny GEMM(M<=5 的 decode 路径,含 bf16 magic-number 反量化)
csrc/rocm/ops.h: ROCm 侧 op 声明/接入
benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py: 新增,按 batch_size 1-4096 对比 torch-fp16 / hybrid-w4a16 / hybrid-w4a16-zp 三 provider,内置 Qwen3-4B、Qwen2.5-7B、SmolLM2-1.7B、Qwen3-8B 权重形状
```

注:pack 末尾标注 9 个文件因超长被丢弃,Triton fused dequant GEMM 的 Python 实现未出现在保留的 diff 中;上述路径划分与 tile heuristic 来自 PR 正文描述。

新增的 benchmark 脚本用 `@triton.testing.perf_report` 对 batch_size 1 到 4096 逐个测 `torch-fp16`、`hybrid-w4a16`、`hybrid-w4a16-zp` 三个 provider,以 `triton.testing.do_bench_cudagraph` 取分位数,ylabel 为 TFLOP/s;`WEIGHT_SHAPES` 内置 Qwen3-4B、Qwen2.5-7B-Instruct、SmolLM2-1.7B-Instruct-AWQ、Qwen3-8B 四组真实权重形状,`prepare_hybrid_weights` 返回 `w_q_skinny` / `w_s_skinny` / `w_fp16` / `w_zp` 四类张量,并注明 Triton 路径直接从 `w_q_skinny` 取 int32 视图、不另开 int32 buffer。

关键代码(benchmark 中 packed shuffle 权重布局 `[N, K//8]`):

```python
def prepare_hybrid_weights(K, N, group_size, device="cuda"):
    """Create random weights for benchmarking.

    Returns (w_q_skinny, w_s_skinny, w_fp16, w_zp). The triton path derives
    its int32 view from w_q_skinny, so no separate int32 buffer is returned.
    """
    num_groups = K // group_size

    # Random packed weights — actual values don't matter for throughput
    w_q_skinny_i32 = torch.randint(
        0, 2**31, (N, K // 8), dtype=torch.int32, device=device
```

## 性能声明

Benchmarked on AMD Strix Halo(gfx1151、LPDDR5X-8000 128 GB),`--max-num-seqs 1`,in/out = 128/128 与 1920/128,指标 TTFT 与 TPOT(ms):`RedHatAI/Qwen3-4B-quantized.w4a16`(compressed-tensors,symmetric)TTFT 82.33 → 68(1.21x)/ 1222.24 → 760(1.61x),TPOT 54.81 → 13.69(4.00x)/ 56.16 → 15.29(3.67x);`Qwen/Qwen2.5-7B-Instruct-AWQ`(AWQ,asymmetric)TTFT 118.64 → 100.09(1.19x)/ 2004.68 → 1183.48(1.69x),TPOT 82.04 → 21.03(3.90x)/ 84.69 → 21.67(3.91x);`RedHatAI/gemma-3-4b-it-quantized.w4a16`(symmetric)TTFT 85.18 → 66(1.29x)/ 1392.00 → 727(1.91x),TPOT 53.39 → 15.59(3.43x)/ 61.36 → 24.24(2.53x);`RedHatAI/Qwen3-8B-quantized.w4a16`(asymmetric)TTFT 150.51 → 140(1.08x)/ 3181.00 → 1529(2.08x),TPOT 116.99 → 24.55(4.77x)/ 119.72 → 26.12(4.58x);`trymirai/SmolLM2-1.7B-Instruct-AWQ`(asymmetric)TTFT 39.02 → 46(0.85x,变慢)/ 769.37 → 413(1.86x),TPOT 22.95 → 7.25(3.17x)/ 26.07 → 10.53(2.48x)。Accuracy(`lm_eval` gsm8k,200 samples):`RedHatAI/Qwen3-4B-quantized.w4a16` 0.835 → 0.855(no regression)。

## 验证与回归风险

PR 正文给出上述 5 个模型的 TTFT/TPOT 表与 1 项 gsm8k 精度(200 samples),benchmark 环境为 `--max-num-seqs 1`,指标仅报 TTFT 与 TPOT、未给吞吐 tok/s;正文未给出 pytest 单元测试命令或 CI 步骤,也未提供除 gsm8k 之外的精度结果。风险面:新增 ROCm gfx11/gfx12 专用 kernel 与按 M 的路由(M<=5 / >5),tile heuristic 仅在 gfx1151/gfx1201 基于 Qwen3-4B、Llama-3.1-8B-AWQ 形状调优;正文显示 SmolLM2-1.7B 在 in=128,out=128 的 TTFT 为 0.85x(变慢),其余指标改善;对称与非对称量化、group size 32/64/128 的组合未在正文中逐一列出验证。

## 关联

无(PR 正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
