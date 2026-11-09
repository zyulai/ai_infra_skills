# PR #32619: [Perf] Create TMA-aligned input scale tensor for DeepGemm on Hopper

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-01-22 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/32619 |

## 一句话结论

直接构造 TMA-aligned input scale 省去 transpose_fp32，DeepSeek-R1 e2e 输出吞吐 +3%。

## 动机

正文指出在 H200 上，每次 `deep_gemm::sm90_fp8_gemm_1d2d_impl` 之前都会启动 `deep_gemm::transpose_fp32`：因为 DeepGEMM 在 Hopper 上要求 input scale tensor 为 column major TMA-aligned layout，若布局不满足，DeepGEMM 就会额外 launch 转置 kernel（profiling 显示额外耗时 1.4 µs）。

## 优化点

在 `fp8_utils.py` 的 `per_token_group_quant_fp8` 中新增 `tma_aligned_scales` 参数：为 true 时直接按 shape `(*batch, m, sf_k)`、stride `(tma_aligned_m * sf_k, 1, tma_aligned_m)` 创建 `x_s` scale tensor，使其从一开始就满足 DeepGEMM 的布局要求，从而免去每次的 `transpose_fp32` kernel。下游 `per_token_group_fp8_quant` kernel 会依据 `scale_num_rows` 与 `scale_stride` 写 scale 值，因此该 kernel 无需改动；同时从 `benchmark_fp8_block_dense_gemm.py` 移除已不需要的 `get_col_major_tma_aligned_tensor`。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/fp8_utils.py: per_token_group_quant_fp8 增加 tma_aligned_scales 参数并构造 TMA-aligned x_s
vllm/model_executor/layers/quantization/input_quant_fp8.py: input quant 侧接入该参数
vllm/utils/deep_gemm.py: DeepGemm 相关工具改动
tests/kernels/quantization/test_per_token_group_quant.py 与 test_block_fp8.py: 新增测试用例
benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py: 移除 get_col_major_tma_aligned_tensor
```

## 性能声明

正文声明 "e2e output token throughput improves 3%"。Benchmark 条件：`vllm serve deepseek-ai/DeepSeek-R1-0528 --tensor-parallel-size 8 --max-num-seqs 8 --trust-remote-code --compilation_config '{"compile_sizes": [1,2,4,8]}' --no-enable-prefix-caching`，配 `vllm bench serve --dataset-name sharegpt --sharegpt-output-len 300 --max-concurrency 8 --num-prompts 1000 --num-warmups 20 --ignore-eos`，1000 个请求全部成功。结果：Output token throughput 449.40 → 464.59 tok/s，Total token throughput 793.06 → 819.87 tok/s，Request throughput 1.50 → 1.55 req/s，Mean TTFT 237.58 → 226.28 ms，Mean TPOT 17.07 → 16.52 ms，Benchmark duration 667.56 → 645.73 s。Profiling：main 有 `deep_gemm::transpose_fp32`（额外 1.4 µs），PR 无该 kernel。Accuracy（gsm8k_eval.py）：Accuracy 0.954 → 0.955、Invalid 0.000 不变。硬件 H200（profiling）；精度为 FP8 block quant / DeepGemm。

## 验证与回归风险

Test Plan 为 `pytest -s -v tests/kernels/quantization/test_per_token_group_quant.py` 与 `pytest -s -v tests/kernels/quantization/test_block_fp8.py`，Test Result 为 "Unit tests passed"，并在两个测试文件中新增用例；另做 gsm8k accuracy 与 profiling。PR 未说明对其他模型、硬件或精度的回归评估；改动影响 FP8 量化 scale tensor 的布局约定。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
