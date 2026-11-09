# PR #59753: [Perf][Qwen4Exp] Add SM121 TP=1 skinny-GEMM plans

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stecasta |
| 合并 | 2026-10-02 |
| 分类 | Speculative Decoding \| tier: major |
| Label | performance, ready, qwen, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/59753 |

## 一句话结论

SM121 TP=1 skinny-GEMM plans 让 Qwen3.8-Flash-Next 单 DGX Spark 并发 1 的 ITL 降低 9.5%。

## 动机

PR 正文 Overview：Qwen3.8-Flash-Next decode ITL 在单台 DGX Spark（SM121、TP=1）并发 1 下降低 9.5%，做法是为当前会回退到 cuBLAS 的 TP=1 BF16 shapes 增加 SM121 skinny-GEMM plans。其中 LM head `(248320, 2560)` 贡献最大（单独 -8.3%），其余行在 `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` 上再贡献 -1.3%（router、HC projections、shared-expert gate 为 BF16，其余 dense projections 为 MXFP8）；`nvidia/Qwen3.8-Flash-Next-NVFP4` 的 attention/GDN/shared experts 也是 BF16 但单台 Spark 装不下该 checkpoint，故未做端到端测量。正文指出 c1 decode step（MTP k=3）中 LM head 跑 4 次：3 个 draft step 在 M=1（cuBLAS gemv 各 6.8 ms）加 target verify 在 M=4（5.3 ms），合计 25.7 ms、约占 77.6 ms step 的 33%。

## 优化点

沿用 SM100/SM90 的表格机制（精确 capability `(12, 1)`、按 local weight shape 键控、config 由 CUDA graph sweep 对 cuBLAS 选出），为 SM121 新增 TP=1 的 `QWEN4_EXP_SM121_GEMM_PLANS`：按 (N,K) shape × num_tokens(1/2/4/8/16) 给出 `SkinnyGemmConfig`，覆盖 LM head (248320,2560)、shared-expert gate、GDN B/A、HC、router、QSA、shared-expert down/up 等 BF16 shapes；新增 `_is_sm121()` 并在 `_gemm_plans()` 中按 capability (12,1) 返回该表。M<=16 走 skinny GEMM，M=32 仍在 cuBLAS。`tests/kernels/test_bf16_skinny_gemm.py` 新增 `QWEN4_EXP_SM121_CASES` 与 `test_qwen4_exp_sm121_selected_shapes`（cosine > 0.999）。正文明确 "This PR does not change any other GPU or the selection logic"。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/low_latency_gemm.py: 新增 QWEN4_EXP_SM121_GEMM_PLANS 计划表、_is_sm121() 与 _gemm_plans() 分支
tests/kernels/test_bf16_skinny_gemm.py: 新增 SM121 能力路由与选中 shape 的 cosine 正确性测试
```

```text
+# DGX Spark (GB10) plans, TP=1.
+QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
+    # LM head.
+    (248320, 2560): {
+        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
...
+def _is_sm121() -> bool:
+    return current_platform.is_device_capability((12, 1))
+    if _is_sm121():
+        return QWEN4_EXP_SM121_GEMM_PLANS
```

## 性能声明

Kernel（GB10 p4242-0063 约 2.5 GHz、FlashInfer CUPTI、CUDA graph、cold L2、median of 30；每个 config 与 `F.linear` 的 cosine 1.000000）：M=1 cuBLAS 7262 → skinny 5016 us（1.45x）、M=2 5217 → 5038（1.04x）、M=4 5257 → 4917（1.07x）、M=8 5291 → 4890（1.08x）、M=16 5336 → 4933（1.08x）、M=32 5608 us / n/a (M <= 16)。正文："cuBLAS reaches about 175 GB/s at M=1 on this shape; the skinny GEMM reaches 253 GB/s: 1.45x at M=1, 1.04x to 1.08x at M=2 to 16"。End to end（node p4242-0074、`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`、TP=1、MTP k=3 forced acceptance length 3、FP8 KV、`--max-num-batched-tokens 4096`、SPEED-Bench 2K in / 1K out、two interleaved runs per arm, mean）：并发 1 ITL p50 25.30 / 23.20 / 22.91 ms（-9.5%，base / LM-head rows only / full table）、output tok/s 38.3 / 41.55 / 42.1（+9.9%）、TTFT p50 891 / 878 / 881 ms；并发 8 ITL 57.92 / 58.05 / 58.17、output tok/s 127.7 / 128.0 / 128.3、TTFT 4429 / 4416 / 4430（parity，其 M=32 verify step 超出 skinny GEMM 范围仍走 cuBLAS）。Accuracy：GSM8K 前 256 题、greedy、thinking off、speculative decoding off、8 并发、每臂两次：base 249/247、LM-head rows only 245/247、full table 246/246（of 256）。Bake-off 注明 M=1 时 b12x `bf16_vocab_projection` 默认 config 比 skinny config 快 4%，但会引入新依赖。

## 验证与回归风险

Tests：`pytest tests/kernels/test_bf16_skinny_gemm.py -k "qwen4_exp" -v` 在 GB10 (p4242-0070) 上跑 Qwen4Exp capability-routing test 与 SM121 selected-shape test（每个 shape/M 的 cosine > 0.999），64 passed；该文件其余用例需要 Spark serving image 里没有的 GLM-5.2 modules。Disclaimers：E2E 跑在 vLLM `af7f9488c` 上并把本 PR 表格覆盖到 `low_latency_gemm.py`；每格请求数少（8 与 16），单节点 baseline 重复差异在 1.5% 内；SM120（RTX PRO 6000）未覆盖，其 LM head 仅 1.07x-1.09x（cuBLAS 已接近该卡带宽）；`(13312, 2560)` 是未融合的 QSA QKV/gate shape，#59533 落地后需为融合后的 `(13952, 2560)` 补一行。正文明确不改其他 GPU 与选择逻辑。

## 关联

正文："Complements #59632 (SM121 plans for TP=2 shapes, from #59605): this PR only adds TP=1 keys to the same mechanism"；表格机制沿用 #59214（SM100/SM90）；#59533 落地后需补融合 shape；Duplicate check 称 #59632 是唯一 open 的 SM121 plan PR 且仅覆盖 TP=2 shapes。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
