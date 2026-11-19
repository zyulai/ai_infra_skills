# PR #65706: [Perf][GLM-5.3] Add FlashInfer warp split-K GEMMs to the SM107 low-latency table and fix q_b's M=1 fallback

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @csahithi |
| 合并 | 2026-11-17 |
| 分类 | Speculative Decoding(tier: major) |
| Label | ready, deepseek, glm |
| Diff | 总 diffstat 缺失(pack 未提供);Details 注明非测试代码 +177/−55 行 |
| 链接 | https://github.com/vllm-project/vllm/pull/65706 |

## 一句话结论

在 SM107 的 GLM 低延迟 GEMM 表中把 FlashInfer warp split-K BF16 GEMM 加入候选(224 个 (projection, M) 点中 199 个由其服务),对 cuBLAS 几何平均 1.21x;同时修复 `fused_norm_rope` 的 M=1 q 输出行 stride 问题,让 q_b/wq_b 在 M=1 真正命中表行。

## 动机

SM107 表此前只有 CuTe skinny GEMM、`dsv3_fused_a` 与 cuBLAS;skinny GEMM 的中等 M 区间没有更好的 kernel。另一个既有 bug:attention 层把 `q_c` 作为 fused q_a + kv_a 输出的列切片传入,`fused_norm_rope` 用 `torch.empty_like(q_c)` 分配输出,单 token 时该 slice 被视为 dense、保留了行 stride(GLM-5.2/5.3 为 2624,indexer 层 2784),而 GLM 表只收 packed 行,导致 q_b_proj 与 wq_b 在 M=1 回退 cuBLAS(该 bug 先于本 PR 存在)。

## 优化点

1. 新 backend `fi_warp_splitk`:`_WarpSplitKGemm`(VllmJitKernel)按 (N, K, tactic) 注册 JIT warmup compile key,M 动态,在 CUDA graph capture 前编译;直接调 `run_warp_splitk_dense(pdl=True)` 以钉住实测 tactic(`mm_bf16` 不接受 tactic)。
2. 表项重写:`_warp_splitk_rows` 按 per-call timing 把 M 1–32 分配给 warp split-K / CuTe / cuBLAS(224 点中 warp split-K 199、CuTe 17、cuBLAS 8);`USE_TD` 式的 `USE_TD` 不涉及,但 `_WarpSplitKGemm.CompileKey` 记录 tactic。
3. q_b 修复:`fused_norm_rope` 输出改为 `torch.empty_like(q_c, memory_format=torch.contiguous_format)`,一行改动同时修复 q_b_proj 与 wq_b。
4. FlashInfer warp split-K 模块不可导入时这些行回退 cuBLAS。

## 关键改动

```python
# vllm/models/deepseek_v32/common/kernels.py
-        q_c_out = torch.empty_like(q_c)
+        # empty_like's default would keep a one-token column slice's row stride;
+        # the low-latency GEMM tables behind q_b_proj and wq_b only take packed rows.
+        q_c_out = torch.empty_like(q_c, memory_format=torch.contiguous_format)
```

```python
# vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py
+    if backend == "fi_warp_splitk":
+        assert isinstance(config, tuple)
+        return _run_warp_splitk(x, weight, config)
```

## 性能声明

数字均照抄 PR 正文(未独立核实):

- 逐 kernel(isolated, VR200, CUDA graph、cold L2、PDL 开):over the 216 planned points, the table is 1.21x faster than cuBLAS(geomean;1.01–2.10x per point);over the 201 points where it differs from main's table,1.16x faster than main's row(geomean;0.99–1.63x)。
- Serving(GLM-5.3-NVFP4, TP4 on one VR NVL72 tray, MTP K7 synthetic acceptance 3.91, 8k in/1k out):+6.9% / +6.8% / +1.8% output throughput at c1 / c2 / c4(c1 median TPOT 2.27 → 2.11 ms),within noise from c8(c8 +0.7%)。
- q_b fix at M=1(VR200, TP4 4096x2048):5.02 → 3.37 µs per call isolated (1.49x),4.13 → 2.98 µs back to back (1.39x)。
- Numerics:worst relative L2 error 1.76e-3 over the 216 planned points,every point's error within 0.001% of cuBLAS。
- Accuracy(earlier revision, GSM8K 8-shot greedy, lossless MTP K7):c4 all 1,319 questions 91.28% → 92.12%(McNemar p 0.25);c1 first 200:89.5% → 90.5%(p 0.82)。

## 验证与回归风险

- `pytest tests/kernels/test_bf16_skinny_gemm.py`:306 passed, 445 skipped, 1 failed(GLM -k glm:142 passed, 97 skipped);新 GPU 测试覆盖每个 planned warp split-K tactic 的最小/最大 M(eager + CUDA graph + 重放前新输入)、warmup 与调用命中同一 FlashInfer cache 条目;CPU 测试校验每个 M 恰一个 backend。
- q_b fix:`tests/kernels/test_fused_deepseek_v32_norm_rope.py` 全部 222 tests 通过(node q 打印 226,含一个后来丢弃的 4-case stride test)。
- 回归风险:warp split-K 求和顺序不同,非 bit-exact,greedy 文本可能漂移;q_b fix 使 SM100/SM103 的 q_b/wq_b M=1 也改走 CuTe(未在那些 GPU 上端到端测量);PDL 内存序问题依赖上游 flashinfer#8081(vLLM 当前 pin 0.7.4 不含该修复);`fused_norm_rope` 是 DeepSeek-V3.2 与 ROCm attention 共用代码。

## 关联

MTP `eh_proj` 行依赖 #65704 先合并;#62843 是 main 的 CuTe 行;#66030 在 Kimi K3.1 特化 decode 中引入了同一 kernel 的适配副本;#60027 曾为 Qwen4Exp 放宽 `_runtime_ok`。对照 commit:`7f07ac6556` / `7f2d731097` / `87879f6f37` / `56fd4cb5a1`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
