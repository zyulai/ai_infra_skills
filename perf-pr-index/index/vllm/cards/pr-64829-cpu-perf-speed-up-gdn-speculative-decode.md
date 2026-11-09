# PR #64829: [CPU][Perf] Speed up GDN speculative-decode recurrent update

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ekintel |
| 合并 | 2026-11-07 |
| 分类 | Speculative Decoding（tier: major） |
| Label | performance, cpu, verified |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 3 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/64829 |

## 一句话结论

CPU GDN spec-decode recurrent kernel 重构，GCP c4-standard-48 / 30 层 8 tokens / fp32 state 由 9.01→4.91 ms（−45%）。

## 动机

正文 Purpose：`fused_sigmoid_gating_delta_rule_update_spec_cpu` 在 CPU 上 speculative-decode verify 的每个 GDN 层都会运行（例如 Qwen3.5/3.6 配 DFlash/MTP），其开销直接叠加到每一步 decode。PR 针对该 kernel 的并行划分与访存模式做优化，并新增对应 benchmark 与测试。

## 优化点

工作划分从 (sequence, value head) 改为 (sequence, value head, 64-column block)：batch 1 时 Qwen3.6 的工作项从 32 增到 64（diff 注释也把并行维度从 `(sequence, v_head)` 改为 `(sequence, v_head, value-column block)`，每个序列的 token 循环因 recurrence 仍保持顺序）。每个工作项只加载一次 fp32 state block，并在全部 draft tokens 间把它保留在 cache；旧 kernel 每个 token 会把前一个 slot 重读两次。仅在后续 step 才会再读的 per-token 中间状态改用 streaming stores（AVX-512、对齐地址）写入，避开每次 store 的 read-for-ownership；diff 中新增 `stream_state_vec2()`，在地址 64 字节对齐时用 `_mm512_stream_ps` / `_mm512_stream_si512`，否则回落到 `store_state_vec2`。

## 关键改动

```text
csrc/cpu/sgl-kernels/fla.cpp: GDN spec recurrent update kernel 的并行划分与访存改写，新增 stream_state_vec2 流式写
benchmarks/kernels/cpu/benchmark_cpu_gdn_spec.py: 新增 cold-state、30 层的 CPU GDN spec kernel benchmark
tests/kernels/mamba/cpu/test_cpu_gdn_ops.py: 新增 spec-kernel 测试（含 unaligned state cache 用例）
```

```diff
+// Like store_state_vec2, but bypasses the cache when possible: for state that
+// will not be read again until a later step, this saves the RFO read.
+template <typename state_t>
+inline void stream_state_vec2(
+    state_t* __restrict__ dst,
+    const at::vec::Vectorized<float>& src0,
+    const at::vec::Vectorized<float>& src1) {
+#if defined(CPU_CAPABILITY_AVX512)
+  if ((reinterpret_cast<std::uintptr_t>(dst) & 63) == 0) {
+    if constexpr (std::is_same_v<state_t, float>) {
+      _mm512_stream_ps(dst, __m512(src0));
+      _mm512_stream_ps(dst + 16, __m512(src1));
```

## 性能声明

测试条件：GCP c4-standard-48，23 threads。Kernel, 30 layers, 8 tokens：fp32 state `9.01 → 4.91 ms (−45%)`；bf16 state `4.90 → 2.61 ms (−47%)`。End-to-end，Qwen3.6-35B-A3B + DFlash2（7 speculative tokens），HumanEval，concurrency 1：median ITL `127.85 → 120.08 ms (−6.1%)`、mean TPOT `23.23 → 22.07 ms (−5.0%)`、output throughput `38.17 → 39.92 tok/s (+4.6%)`，Acceptance length `5.58 vs 5.53`。Kernel batch scaling（fp32 state, 8 tokens per request, 30 layers）：

| batch | before | after |
|---|---|---|
| 1 | 8.97 ms | 5.45 ms (−39%) |
| 4 | 28.60 ms | 20.77 ms (−27%) |
| 16 | 110.86 ms | 82.89 ms (−25%) |

## 验证与回归风险

验证：新增 `test_fused_sigmoid_gating_delta_rule_update_spec_cpu`（fp32/bf16/fp16 state；uneven、empty、0-accepted rows；value dim 非向量宽度整数倍），已有 GDN CPU 测试全通过；`pytest tests/kernels/mamba/cpu/test_cpu_gdn_ops.py`：224 passed（含新 spec-kernel 的 unaligned state cache 用例）；`pytest tests/v1/e2e/test_cpu_spec_decode.py -k mtp`：Qwen3.5-0.8B (GDN) with MTP，1 passed；gsm8k（5-shot, limit 200）通过完整 DFlash2 + Qwen3.6-35B-A3B server：`0.290 after vs 0.295 before`（±0.032 standard error）。回归风险：正文明确 "Outputs are not bit-identical"，因求和顺序改变，且 bf16/fp16 state 下 running state 在 draft tokens 间保持 fp32；PR 未注明其他受影响路径。

正文 Test plan 还列出新测试覆盖 fp32/bf16/fp16 state、uneven 与 empty 及 0-accepted rows、value dim 非向量宽度整数倍等情形，并说明 "All existing GDN CPU tests pass"；新增的 `benchmarks/kernels/cpu/benchmark_cpu_gdn_spec.py` 为 cold-state、30 层基准，脚本注释指出 serving 中同一层 state 两次访问之间约有 10 GB 权重流过、state 总是 cold，故通过远大于 LLC 的 footprint 轮转来复现 DRAM 命中，并按 `bytes_per_layer = state_bytes * (1 + q_len)` 估算带宽。正文结尾另注明 "Assisted by GHCP"。

## 关联

无（PR 正文与 diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
