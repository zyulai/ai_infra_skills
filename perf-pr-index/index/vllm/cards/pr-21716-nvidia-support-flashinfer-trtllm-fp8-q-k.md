# PR #21716: [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv/out Attention Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-08-19 |
| 分类 | Quantization \| tier: major |
| Label | performance, rocm, ready, ci/build, v1, llama |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 7 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/21716 |

## 一句话结论

接入 FlashInfer TRTLLM FP8 attention kernel，fp8 + enable_attn_fusion 下达 3.21 req/s（main fp8 为 2.79）。

## 动机

PR 正文 Purpose 给出两点：一是重构 fusion_attn pass 中的 `AttentionStaticQuantPattern`，把 attn + fp8_quant pattern 融合起来，以便使用 TRTLLM FP8-in FP8-out kernel；二是与其他 backend 对齐——当 kv cache dtype 设为 FP8 时，FlashInfer 需要为 query 插入 FP8 quant op。此前这一融合模式与 query 量化路径未打通，FlashInfer 无法走 q/kv/out 全 FP8 的 TRTLLM attention kernel。

## 优化点

在编译期 fusion_attn pass 中改写 `AttentionStaticQuantPattern`，使 attention 与后续 fp8 quant 融合为 TRTLLM FP8 内核可消费的模式；FlashInfer backend 在 kv cache 为 FP8 时对 query 插入量化 op，从而 q/kv/out 全部以 FP8 参与 TRTLLM prefill/decode kernel。配套的 decode、prefill benchmark 脚本被重构为以 `(q_quant_dtype, kv_quant_dtype, o_quant_dtype)` 三元组参数化并引入 `FP8_DTYPE`；单元测试 `test_flashinfer_trtllm_attention.py` 用 `QUANT_DTYPES` 覆盖 `(None, None, None)` 与 `(FP8_DTYPE, FP8_DTYPE, FP8_DTYPE)` 两种组合。素材保留的 diff 仅含这 3 个 benchmark/test 文件，fusion_attn pass 与 FlashInfer backend 的源码改动位于被丢弃的 7 个文件中，素材未提供其 diff。

## 关键改动

```text
benchmarks/kernels/benchmark_trtllm_decode_attention.py: benchmark_decode 改为接收 quant_dtypes 三元组与 batch/heads 参数，引入 FP8_DTYPE
benchmarks/kernels/benchmark_trtllm_prefill_attention.py: 为 prefill kernel 增加同样的 FP8 量化参数化
tests/kernels/attention/test_flashinfer_trtllm_attention.py: 用 QUANT_DTYPES 覆盖 q/kv/out 全 FP8 组合
```

```diff
-FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
-
-# KV Cache Layout for TRT-LLM
-# kv_cache_shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)
+FP8_DTYPE = current_platform.fp8_dtype()
+QUANT_DTYPES = [
+    # (q_quant_dtype, kv_quant_dtype, o_quant_dtype)
+    (None, None, None),
+    (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
+]
```

## 性能声明

E2E 条件：模型 `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8`，Successful requests 640、Maximum request concurrency 128、Total input tokens 653975、Total generated tokens 655360；硬件型号 PR 未注明测试条件。原文数字（Request throughput req/s / Output token throughput tok/s / Total Token throughput tok/s / Mean TTFT ms / Mean TPOT ms）：main `auto` 3.05 / 3124.98 / 6243.37 / 1367.30 / 39.61；main `fp8` 2.79 / 2860.98 / 5715.91 / 1413.01 / 43.35；PR `auto` 3.00 / 3074.24 / 6141.98 / 1375.34 / 40.28；PR `fp8` 2.77 / 2836.64 / 5667.28 / 1439.79 / 43.71；PR `fp8, enable_attn_fusion` 3.21 / 3284.02 / 6561.10 / 1347.81 / 37.65。gsm8k 5-shot flexible-extract：main auto 0.9181、main fp8 0.9234、PR fp8+fusion 0.9128；strict-match 0.8999 / 0.8999 / 0.8923。以上为不同配置横向对比，正文未给出严格同配置 before/after 增益结论。

## 验证与回归风险

Functional：`tests/kernels/attention/test_flashinfer_trtllm_attention.py` 给出 `192 passed, 10 warnings in 143.25s (0:02:23)`；`tests/compile/test_fusion_attn.py::test_attention_quant_pattern` 给出 `6 passed, 4 warnings in 33.01s`。E2E 用 gsm8k 精度确认量化未显著掉点。回归风险：改动涉及 FP8 kv cache + attn fusion 组合路径，可能影响其它使用 FlashInfer TRTLLM attention 的模型与精度（gsm8k strict-match 由 0.8999 变为 0.8923/0.9007），PR 未注明更多受影响路径或硬件。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
