# PR #23647: [Flashinfer] Support Flashinfer TRTLLM FP8-qkv BF16/FP16-out Attention Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-09-09 |
| 分类 | Quantization |
| Label | performance, ready, v1 |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/23647 |

## 一句话结论

Flashinfer + fp8 KV cache 下把 query 也量化为 fp8 并走 TRTLLM attention kernel,decode 内核最高提速约 50%。

## 动机

此前 `Flashinfer + kv_cache_dtype=fp8` 只有在开启 attn fusion 时才把 query 量化为 fp8(`q_data_type` 默认跟随 model dtype),fp8 query + bf16/fp16 输出的 TRTLLM kernel 组合(FP8-qkv BF16/FP16/FP8/NVFP4-out)不可用;query 以高精度参与 attention 意味着额外的 QK 量化/转换开销与更大的访存。

## 优化点

后端层面把 fp8 KV cache 下的路径统一起来:`q_data_type` 直接取 `kv_cache_dtype`(不再依赖 `enable_attn_fusion`),FP8 query 量化(`ops.scaled_fp8_quant`)从 fusion 分支移到 `attn_metadata.q_data_type == FP8_DTYPE` 时无条件执行,并删掉"query 为 FP8 必须已发生 output fusion"的 assert;非 fp8 KV cache 增加 dtype 一致性 assert。TRTLLM kernel 调用支持 BF16/FP16 输出(以及既有的 FP8/NVFP4 输出)。配套:decode/prefill benchmark 与 kernel 单测加入 `(fp8, fp8, None)` 量化组合与 `window_left`(sliding window)参数化;`fusion_attn.py` 的 NVFP4 quant pattern 注册增加 CUDA 平台与 `scaled_fp4_quant` op 存在性检查。依赖 Flashinfer 0.3.0。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: q_data_type=kv_cache_dtype,FP8 query 量化无条件化
vllm/compilation/fusion_attn.py: NVFP4 pattern 注册加平台/op 守卫
tests/kernels/attention/test_flashinfer_trtllm_attention.py: 新增 fp8-q/bf16-out 组合与 window_left 参数化
```

## 性能声明

kernel decode 基准(`benchmark_trtllm_decode_attention.py`,q fp8/kv fp8/out bf16,速度为相对 baseline 的降低比例;GPU 未注明):batch 1/seq 1024 为 0.222(-22%),batch 256/seq 1024 为 0.391,batch 128/seq 32768 为 0.509,batch 256/seq 65536 为 0.472(2.308 vs 4.373 ms),batch 128/seq 131072 为 0.522;小 batch 长 seq 约 0.16-0.40。prefill 基准:seq 8192-32768 各 batch 约 0.14-0.30;batch 128/256 @ seq 2048 约 -0.003(基本持平略负)。质量:lm_eval gsm8k 5-shot 500 题,nvidia/Llama-3.3-70B-Instruct-FP8:PR 下 kv fp8 0.938/0.904(flexible/strict)与 kv auto 持平;fp4 checkpoint kv fp8 0.938/0.862。硬件未注明。

## 验证与回归风险

`test_flashinfer_trtllm_attention.py` 112 passed / 8 skipped(新增 fp8-q 组合与 window_left 参数化);`test_fusion_attn.py::test_attention_quant_pattern` 12 passed。风险:fp8 query 量化对精度敏感(FP4 checkpoint strict-match 0.850→0.862 有波动);非 fp8 KV cache 路径行为不变但有新 assert;依赖 Flashinfer 0.3.0 的 kernel 可用性。

## 关联

同属 Flashinfer/TRTLLM attention fp8 量化输出系列(FP8/NVFP4-out 组合已在既有 benchmark 矩阵中);与 attn fusion(compilation/fusion_attn.py)共用 enable_attn_fusion 配置面。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
