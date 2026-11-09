# PR #22095: [NVIDIA] Support Flashinfer TRT-LLM Prefill Attention Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-08-05 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 2 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/22095 |

## 一句话结论

新增 TRTLLM prefill 并统一 decode/prefill，BF16 kernel 基准 speedup_percent 最高 0.70。

## 动机

此前 #19825 为 decode 路径支持了 TRTLLM attention kernel，本 PR 补齐 prefill 路径，并统一 decode 与 prefill 代码路径，仅用一个环境变量 `VLLM_USE_TRTLLM_ATTENTION` 控制。正文的 checklist 标记 purpose、test plan、test results 三项已完成，仅文档更新一项未勾选。当前 TRTLLM prefill kernel 不支持 Q-BF16/KV-FP8/O-BF16 配置，待 attn+quant fusion 支持后直接用 Q-FP8/KV-FP8/O-FP8/O-NVFP4。正文称 TRTLLM prefill kernel 性能「seems to be much better than the original kernels」。

## 优化点

把 `VLLM_USE_TRTLLM_DECODE_ATTENTION` 统一改名为 `VLLM_USE_TRTLLM_ATTENTION`，decode 与 prefill 共用单一开关，原先按路径区分的判定与日志一并改名。在 flashinfer 后端增加 prefill 路径的 TRTLLM kernel 调用，新增 `benchmarks/kernels/benchmark_trtllm_prefill_attention.py` 基准；测试由 decode 专用的 `test_flashinfer_trtllm_decode_attention.py` 扩展为 `test_flashinfer_trtllm_attention.py`，并补充 `vllm/utils/flashinfer.py` 的工具函数。正文把「对比集成前后性能、lm_eval 精度、kernel 级精度与基准」列为测试计划。

## 关键改动

```text
vllm/envs.py、vllm/v1/attention/backends/flashinfer.py: 把 decode 专用 env 统一为 VLLM_USE_TRTLLM_ATTENTION
vllm/attention/backends/flashinfer.py、vllm/utils/flashinfer.py: prefill 路径接入 TRTLLM kernel
benchmarks/kernels/benchmark_trtllm_prefill_attention.py: 新增 prefill kernel 基准
tests/kernels/attention/test_flashinfer_trtllm_attention.py: prefill/decode 测试
```

```diff
-    "VLLM_USE_TRTLLM_DECODE_ATTENTION":
-    lambda: os.getenv("VLLM_USE_TRTLLM_DECODE_ATTENTION", None),
+    "VLLM_USE_TRTLLM_ATTENTION":
+    lambda: os.getenv("VLLM_USE_TRTLLM_ATTENTION", None),
```

## 性能声明

Kernel 基准 `benchmarks/kernels/benchmark_trtllm_prefill_attention.py`：q_dtype=bfloat16、kv_cache_dtype=bfloat16、output_dtype=bfloat16，PR 未注明 GPU 型号；workload 为 num_seqs 1–256 × max_seq_len 2048–131072，给出 trt_mean、baseline_mean、speedup_percent。该基准覆盖 num_seqs 1 至 256、序列长度 2048 至 131072 的多种组合，speedup_percent 大致落在 0.249 到 0.695 之间，长序列与高并发下仍保持优势。示例：num_seqs=1/2048 trt 0.102 vs baseline 0.233（0.562）；1/4096 0.259 vs 0.699（0.629）；1/131072 177.47513 vs 548.54210（0.67646）；256/2048 8.186 vs 10.907（0.249）；65536/256、131072/128 等为 OOM。lm_eval：Llama-4-Scout-17B-16E-Instruct-FP8、modelopt、kv_cache_dtype auto、TP1、compile level 3（+rms_norm、full_cuda_graph）、max_model_len 2048、limit 500、5-shot、batch 200；gsm8k flexible-extract exact_match 0.936±0.0110、strict-match 0.914±0.0126。

## 验证与回归风险

`tests/kernels/attention/test_flashinfer_trtllm_attention.py`：160 passed, 32 skipped, 129 warnings in 26.46s。lm_eval 结果见性能声明。回归风险：当前 TRTLLM prefill kernel 不支持 Q-BF16/KV-FP8/O-BF16 配置（待 attn+quant fusion 支持后改用 Q-FP8/KV-FP8/O-FP8/O-NVFP4）；超大 shape（如 65536/256、131072/128）出现 OOM。

## 关联

正文明确前序 PR 为 #19825（decode 路径的 TRTLLM attention kernel）；未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
