# PR #62219: [Kernel][Perf] Add H100 tuned configs for Llama-3.1-8B and Qwen3-8B to the batch-invariant matmul table

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Monishver11 |
| 合并 | 2026-10-26 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, llama, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/62219 |

## 一句话结论

H100 SXM 上为 BI matmul 补 8 个 tile 配置，`vllm bench latency` 比 main 快 1.85–2.79x。

## 动机

Overview：在 batch-invariant（BI）模式（`VLLM_BATCH_INVARIANT=1`）下，每个 bf16 linear 层与 lm_head 都跑 Triton kernel `matmul_kernel_persistent` 而非 cuBLAS，其 tile 来自 per-GPU 表；Hopper 表只有 5 个 Qwen3-1.7B 形状（在 H20 上测得），因此 Llama-3.1-8B 与 Qwen3-8B 在 H100 上只能使用默认 128x128x64 tile，带来明显的 BI tax。本 PR 为这两个模型的 8 个 TP=1 形状补上 H100 实测的 `"hopper"` 条目。

## 优化点

在 `_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["hopper"]` 新增 8 个 `(N,K)` 条目：Llama-3.1-8B 与 Qwen3-8B 的 gate_up/down/lm_head 及二者共享的 qkv/o；每个形状单一 `BLOCK_K`，使一行的 K-reduction 顺序不依赖 M（保证 bitwise），现有条目与其他 family 不动。每个形状按 M bucket 给出 `_MatmulMConfig(BLOCK_M, BLOCK_N, num_warps, num_stages)`：小 M 用小 tile（如 16x128 / 16x64）、大 M 用 128x128。经真实 `matmul_persistent` dispatch 逐位校验 2028/2028 一致。PR 说明 Qwen3-8B 在 `vllm bench serve` 无变化，原因是这些形状切换的执行模式而非 tile。

## 关键改动

```text
vllm/model_executor/determinism/batch_invariant_configs.py: 在 "hopper" family 新增 8 个 (N,K) 形状及 M-bucket tile 配置（diff 中仅此 1 个文件）。

  (24576, 4096): _MatmulShapeConfig(  # Qwen3-8B gate_up_proj
      block_k=128,
      m_buckets=(
          (1, _MatmulMConfig(16, 128, 4, 5)),
          (4, _MatmulMConfig(16, 64, 4, 5)),
          ...
          (2048, _MatmulMConfig(128, 128, 8, 3)),
      ),
  ),
```

## 性能声明

测试条件（Setup）：1x H100 80GB HBM3 SXM、132 SMs、700 W；driver 595.91.07、CUDA 13.0、torch 2.14.0+cu130、triton 3.8.0；main = `307c5844ad`，PR = main + 本 commit（`git am`），同一 checkout `git checkout` 切换、每次 fresh process；模型 `unsloth/Llama-3.1-8B-Instruct` 与 `Qwen/Qwen3-8B`；`VLLM_BATCH_INVARIANT=1`、`--max-model-len 2048`、batch-size 8。Claims：`vllm bench latency` BI on 比 main 快 1.85–2.79x；BI tax（BI on / BI off）由 main 的 2.13–2.99x 降到 1.03–1.20x；GSM8K 在 0.5 SE 内。E2E latency（BI=0 / BI=1 main / BI=1 PR，单位 s）：Llama-3.1-8B 32in/128out bs8 0.854/2.452/0.881（2.78x，tax 2.87x→1.03x）、1024in/32out 0.403/0.859/0.444（1.94x，2.13x→1.10x）；Qwen3-8B 32in/128out 0.893/2.668/0.956（2.79x，2.99x→1.07x）、1024in/32out 0.414/0.917/0.495（1.85x，2.22x→1.20x）。Kernel（H100 SXM, cold L2，us/call）：80 个 (shape,M) 用例 geomean 2.03x vs main，M<=64 为 2.89x，最慢 1.00x（Qwen3 gate_up M=2048，827 vs 828 us）；M<=64 时内核为 cuBLAS 的 0.90–1.12x（main 为 1.48–4.09x）。代表行 main→PR：Llama gate_up (28672,4096) M=1 236→83(2.85x)、M=2048 983→740(1.33x)；Llama down (4096,14336) M=1 182→50(3.66x)；Qwen3 gate_up (24576,4096) M=1 228→72(3.17x)、M=2048 827→828(1.00x)；qkv (6144,4096) M=1 80→23(3.51x)；o (4096,4096) M=1 69→18(3.77x)；Llama lm_head (128256,4096) M=1 816→347(2.35x)；Qwen3 lm_head (151936,4096) M=1 936→407(2.30x)。Serve（1024in/128out、1000 prompts、rate inf、3 轮交错，req/s）：Llama-3.1-8B 28.32 / 21.64 / 24.59（+13.6%，throughput tax 1.31x→1.15x），mean TPOT 116.4→103.2 ms（BI off 88.4）、mean TTFT 20.41→18.12 s；Qwen3-8B 27.48 / 21.48 / 21.48（0.0%，tax 1.28x→1.28x）。GSM8K（5-shot, BI=1）各差异均在 0.5 SE 内：Llama strict/flexible 0.7180±0.0124/0.7832±0.0114 → 0.7119±0.0125/0.7839±0.0113；Qwen3 0.8764±0.0091/0.8779±0.0090 → 0.8741±0.0091/0.8757±0.0091。

## 验证与回归风险

Bitwise：`matmul_persistent` PR 树全部 13 个 hopper shapes；probe row 在 rows 0、M/2、M-1 须等于 M=1 计算的同行；39 个 M 值（1-17、每个 bucket edge ±1、100、1000、3000、4096）、2 seeds、权重以 `w.t()` 与连续 `(K,N)` 两种布局 → 2028/2028 identical、0 mismatched。Tests：`pytest -v tests/v1/determinism/test_matmul_batch_invariant.py` 35 passed、2 skipped；`VLLM_TEST_MODEL=Qwen/Qwen3-8B pytest -v tests/v1/determinism/test_batch_invariance.py` 22 passed；以 `unsloth/Llama-3.1-8B-Instruct` 同为 22 passed。GSM8K 差异均在 0.5 SE 内。风险与边界：仅 `"hopper"` family 的 8 个新形状，其他 family/条目不动；单一 `BLOCK_K` 保证 bitwise；Qwen3-8B serve 吞吐无变化（执行模式切换而非 tile）；Setup 注明 Llama checkpoint 为 ungated 的 unsloth 副本、未与 gated 权重比对；#61118 在该 base 之后合入、启用 Inductor 的 `batch_invariant` flag，只影响 `torch.compile` arms，不改变本 PR 默认 breakable-graph 模式。

## 关联

PR 正文 Related 明确提及：tracker #27433（issue）、同类型表的 #53247、#57456、#58495；Validation 提及 #61118（在该 base 之后合入，启用 Inductor batch_invariant flag，只影响 torch.compile arms）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
