# PR #43677: [Perf] Optimize Fp8BlockScaledMMLinearKernel input_scale tensor using new_empty()

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-05-27 |
| 分类 | Quantization（tier: major） |
| Label | performance, ready, deepseek |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 1 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/43677 |

## 一句话结论

占位 input_scale 由 new_ones 改 new_empty，省掉一次 vectorized_elementwise_kernel，H200 上 DeepSeek-V4-Pro e2e output tok/s 提升 2%。

## 动机

正文 Purpose：作者在 profiling DeepSeek V4 时发现 `at::native::vectorized_elementwise_kernel` 会在 `tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel` 之前被 launch，原因是在 `apply_input_quant=False` 时，`input_scale` 张量是用 `new_ones()` 创建的（BlockScaledMMLinearKernel.py L129）。

## 优化点

由于 `apply_input_quant=False` 时 `input_scale` 只是占位张量、下游 `apply_block_scaled_mm` 并不使用，PR 把 `input_2d.new_ones(1)` 替换为 `input_2d.new_empty(1)`，去掉对占位张量的初始化写操作，从而不再触发 `at::native::vectorized_elementwise_kernel`。Profiling 段对比：Main 会 launch 该 kernel，PR 分支不会。改动仅一行，属于消除一次冗余 elementwise kernel launch 及其带来的 CPU/GPU 开销。

正文 Purpose 还给出定位链路：作者 profiling DeepSeek V4 时发现 `at::native::vectorized_elementwise_kernel` 排在 `tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel` 之前，回溯到 `BlockScaledMMLinearKernel.py` 的 `apply_weights`（正文引 v0.21.1rc0 的 L129）；并给出下游 `apply_block_scaled_mm` 不使用该占位张量的代码链接（flashinfer.py L126-L139）。改动位于 `@@ -126,7 +126,7 @@` 的单行，仅替换 `apply_input_quant=False` 分支中 `input_scale is not None` 为假时的构造方式，`apply_input_quant=True` 分支与 `apply_block_scaled_mm` 均未改动。Profiling 的 Main/PR 两张截图分别对应存在与不存在该 elementwise kernel 两种情形，正文据此确认冗余 launch 已被消除。

## 关键改动

```text
vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py: apply_weights 中 apply_input_quant=False 分支的占位 input_scale 由 new_ones(1) 改为 new_empty(1)
```

```diff
             input_scale = (
-                input_scale if input_scale is not None else input_2d.new_ones(1)
+                input_scale if input_scale is not None else input_2d.new_empty(1)
             )
```

## 性能声明

测试条件：H200；`vllm serve deepseek-ai/DeepSeek-V4-Pro --tensor-parallel-size 8 --max-num-seqs 8 --kv-cache-dtype fp8 --block-size 256 --max-model-len 800000 --gpu-memory-utilization 0.95 --max-num-batched-tokens 512 --no-enable-flashinfer-autotune --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' --tokenizer-mode deepseek_v4 --no-enable-prefix-caching`；`vllm bench serve --dataset-name sharegpt --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 300 --num-warmups 50 --ignore-eos --temperature 0`（num-prompts / max-concurrency 按并发取值）。

| Metric | Main c=1 | PR c=1 | Main c=8 | PR c=8 |
|---|---|---|---|---|
| Output token throughput (tok/s) | 77.67 | 79.27 | 352.64 | 359.24 |
| Total token throughput (tok/s) | 146.16 | 149.17 | 613.94 | 625.43 |
| Mean TTFT (ms) | 284.41 | 278.29 | 522.20 | 502.99 |
| Mean TPOT (ms) | 11.97 | 11.73 | 21.01 | 20.66 |
| Median ITL (ms) | 11.93 | 11.65 | 16.96 | 16.72 |
| P99 ITL (ms) | 13.73 | 13.49 | 221.41 | 217.60 |

正文结论："e2e output token throughput improves 2% on H200."（perf claims 段同样标注 2%）。

## 验证与回归风险

验证：Profiling 截图对比（Main 有 `at::native::vectorized_elementwise_kernel`，PR 无）；上述 serving benchmark（Main/PR、concurrency 1/8）；Accuracy Testing 用 `python3 -m lm_eval --model local-completions --model_args model=deepseek-ai/DeepSeek-V4-Pro,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 --tasks gsm8k`，Main exact_match 0.9522/0.9522（flexible/strict，±0.0059/0.0059），PR 0.9553/0.9560（±0.0057/0.0056）。可能影响路径：`apply_input_quant=False` 分支下 `input_scale` 变为未初始化的占位张量，正文说明该张量下游 `apply_block_scaled_mm` 不使用；PR 未注明其他回归风险。

## 关联

无（PR 正文未给出其他 PR 编号；Purpose 段引用的是源码文件链接而非 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
