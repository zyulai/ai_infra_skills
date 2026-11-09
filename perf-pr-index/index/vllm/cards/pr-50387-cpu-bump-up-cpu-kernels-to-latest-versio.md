# PR #50387: [CPU] Bump up CPU kernels to latest version

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bigPYJ1151 |
| 合并 | 2026-07-30 |
| 分类 | Speculative Decoding（tier: major） |
| Label | cpu |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾注明 "13 file(s) dropped from diff for size (see PR for full diff)" |
| 链接 | https://github.com/vllm-project/vllm/pull/50387 |

## 一句话结论

同步 sglang CPU kernels 并让 GDN prefill 直接读写 pooled ssm_state（省一次拷贝），Qwen3.5 GDN 路径吞吐/TTFT/TPOT 改善 3–8%。

## 动机

正文 Purpose：把 `csrc/cpu/sgl-kernels/` 从 sglang（`sgl-kernel/csrc/cpu`）同步到最新上游 main，并调和自上次 bump（#41924）以来叠加的 vLLM 专属补丁：MXFP4 W4A16 MoE、AMX GDN dispatch、ISA-portability BLAS fallback、RISC-V scalar/RVV 支持、chunked-prefill `has_initial_state` 修复、DFlash speculative decoding、batched spec-decode conv-state updates；同时拒绝了一个上游回归（MXFP4 scale validation 的 `TORCH_CHECK` 会静默削弱 size check）。

## 优化点

采用 SGLang 新的 pooled-state-indexing `chunk_gated_delta_rule_cpu` 签名用于 GDN prefill：CPU GDN 路径现在通过 `initial_state_indices` 直接读写 pooled `ssm_state` buffer，不再先 gather 一份本地拷贝，省掉每次 prefill 调用的一次额外拷贝；并让 kernel 在运行时读取 `initial_state` 实际的 `stride(0)`（对齐 `conv.cpp` 的 `conv_state_slot_stride`），不再假设完全 packed 布局，因为 pool buffer 可能按 slot padding（如 Qwen3.5 的 Mamba cache-align mode）。diff 中 `fla.cpp` 大幅重写（含 AMX CPU 上的 GDN 优化注释：融合 `kkt_solve`/`recompute_w_u`、VNNI pack 等），`conv.cpp` 的 silu 改用 `fast_silu`，`common.h` 记录 sync commit/date（2026-07-29）并新增 dispatch 宏。

## 关键改动

```text
csrc/cpu/sgl-kernels/fla.cpp: 大规模同步上游（AMX GDN 优化、VNNI pack、pooled-state indexing），该 diff 由 929 行改为 1550 行
csrc/cpu/sgl-kernels/conv.cpp: silu 改用 fast_silu 等同步改动
csrc/cpu/sgl-kernels/common.h: 记录 sync commit/date，新增 AT_DISPATCH_* 宏
其余 13 个文件因体积被 dropped
```

```cpp
// [NOTE] GDN Optimizations on AMX CPU
//   * intra loop: fuse `kkt_solve` and `recompute_w_u` so as to avoid materialize `A`.
//   * inter loop: fuse `recompute_w_u` and `update_v` so as to avoid materialize `h` and `v_new`.
//   * intra loop parallel on H instead of Hv, remove duplicated key @ key.T
//   * fuse format pack with elemwise OP as much as possible.
//   * update state (FP32) with amx-bf16 where C(FP32) += A(BF16) * B(BF16)
//   * compile time mask out upper triangular part in decay mask and tril solve, reduce fma needed.
```

## 性能声明

测试条件（Test Result 段）：Build 三个 CPU ISA targets；`vllm serve` + `vllm bench serve` 前后对比 4 模型（w8a8、w4a16、FP8 MoE、Qwen3.5）；`lm_eval` GSM8K（200 题、5-shot、chat template、max_model_len=16384、max_gen_toks=8192）在 `Qwen/Qwen3.5-35B-A3B`。正文："throughput/latency within noise (±3%) for the three MoE GEMM paths (w8a8, w4a16, FP8); Qwen3.5 GDN path improved 3-8% across throughput/TTFT/TPOT from the pooled-gather optimization"（perf claims 段标注 8% across throughput）；"GSM8K: before `0.795/0.785` (flexible/strict exact_match), after `0.835/0.795` — within one standard error, no regression"。PR 正文未给出 serving benchmark 的逐项 tok/s 数值。

## 验证与回归风险

验证：Build `VLLM_TARGET_DEVICE=cpu python3 setup.py develop`（`_C`、`_C_AVX512`、`_C_AVX2` 三个 target 均干净通过）；正确性：`tests/quantization/test_cpu_wna16.py` / `test_cpu_w8a8.py` 的 12 个模型（AWQ/GPTQ/compressed-tensors W4A16、FP8 W8A16 dense/MoE、MXFP4、INT8 W8A8 MoE）离线推理前后对比，10/12 与 bump 前 greedy 输出逐字节一致，2 个（`gpt-oss-20b` MXFP4、`Qwen3-30B-A3B-GPTQ-Int4`）在首个 token 后发散，正文归因于量化 kernel 数值使近并列 argmax 变化而非功能损坏；`Qwen/Qwen3.5-35B-A3B` 多序列 prefill 与 prefix-cache reuse（`has_initial_state=True`）路径；4 模型 serving benchmark；GSM8K 见性能声明。可能影响路径：全部 CPU kernels（尤其 GDN/AMX、MXFP4、RISC-V、ISA fallback），2 个量化模型输出非逐字节一致；正文称已拒绝一个上游 MXFP4 scale 校验回归。PR 未注明更多回归风险。

## 关联

#41924（正文 "since the last bump (#41924)"，即上一次 CPU kernels bump 的前序 PR）；正文另称 "Not a duplicate of any open PR — searched `gh pr list` for sgl-kernel/CPU-kernel-bump related terms, no matches"，未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
