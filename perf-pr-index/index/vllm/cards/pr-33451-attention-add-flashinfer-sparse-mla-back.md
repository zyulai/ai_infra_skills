# PR #33451: [Attention] Add FlashInfer Sparse MLA backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-02-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, rocm, ready, ci/build, v1, cpu, nvidia |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 另有 13 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/33451 |

## 一句话结论

集成 FlashInfer sparse MLA kernel 作为 DeepSeek MLA 新 backend,head 数 ≤16(TP8+)时默认启用;16 q heads 的 prefill 上比 FlashMLA_SPARSE 快约 6%-19%。

## 动机

FlashInfer 上游(flashinfer PR #2138)已支持 sparse MLA,而 vLLM 此前只有 FLASHMLA_SPARSE 一个 sparse MLA backend。该 PR 把这一 kernel 接入 vLLM,并在 head count ≤16(TP8 或更高)时默认启用,补齐 DeepSeek 系列在大 TP 分片下 prefill 路径的 backend 选择与性能。

## 优化点

新增 `FLASHINFER_MLA_SPARSE` backend 并注册进 `AttentionBackendEnum`(设计文档优先级表列为 7),sparse 路径通过 indexer 的 `topk_indices_buffer` 决定每个 token attend 哪些 KV slot,decode 走 `forward_mqa`。配套改造 attention benchmark 框架:backend 解析从"模块路径+类名命名约定"改为经 `AttentionBackendEnum.get_class()` 取真实 backend 类(含 `get_supported_kernel_block_sizes`、`is_sparse`),新增 `MockIndexer`(分配并随机填充 topk_indices_buffer)与 `index_topk`(默认 2048)支持,新增 `mla_prefill.yaml` 用 `model_parameter_sweep` 以 num_q_heads ∈ {128, 64, 32, 16} 模拟 TP1-8。正确性测试改为按各 backend 声明的 `supported_kv_cache_dtypes`(auto/fp8/fp8_ds_mla)与支持 block size(32/64)参数化,并按平台 capability skip(FlashInfer 要求 SM 10.0+)。

## 关键改动

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py: 新 sparse MLA backend(diff 被截断,路径取自测试 import)
tests/v1/attention/test_sparse_mla_backends.py: FlashMLA/FlashInfer sparse decode 正确性参数化测试
benchmarks/attention_benchmarks/{common,mla_runner,runner}.py: benchmark 改用 AttentionBackendEnum 解析,新增 MockIndexer 与 sparse 支持
benchmarks/attention_benchmarks/configs/mla_prefill.yaml: 新增 prefill 配置(TP 模拟 sweep)
docs/design/attention_backends.md: backend 支持矩阵与优先级表新增 FLASHINFER_MLA_SPARSE
```

## 性能声明

用 `benchmarks/attention_benchmarks/benchmark.py --config ...mla_prefill.yaml` 对比 FLASHMLA_SPARSE 与 FLASHINFER_MLA_SPARSE:DeepSeek-V3 MLA 维度(60 层、head_dim 576、kv_lora_rank 512)、bf16、index_topk 默认 2048、prefill/extend batch specs;硬件 PR 未注明。num_q_heads=16(模拟 TP8):FlashInfer 在全部 19 个测得 spec 上最优,FlashMLA_SPARSE 相对慢 6.3%-18.8%(例:1q1k 0.000419 vs 0.000361 s;8q1k 0.003175 vs 0.002673 s)。num_q_heads=32:FlashInfer 时间为 FlashMLA 的约 155%-158%;64 heads:约 275%-285%;128 heads:约 127%-131%(FlashMLA 均为最优)。部分大 batch spec(2q8k、4q4k、4q8k、8q2k/4k/8k)两个 backend 均报 ERROR。正确性:serve DeepSeek-V3.2 `-tp 8 -ep` 后 lm_eval gsm8k(5-shot)exact_match flexible-extract 0.9591 / strict-match 0.9583。

## 验证与回归风险

`tests/v1/attention/test_sparse_mla_backends.py` 扩展为 backend(FlashMLA/FlashInfer)× kv_cache_dtype(auto/fp8/fp8_ds_mla,按 backend 支持跳过)× tensor_parallel_size(1/2/4)× block_size(32/64)的参数化 decode 正确性测试;FlashInfer sparse 需 SM 10.0+,FlashMLA 走 `is_flashmla_sparse_supported()`。回归影响:head 数 ≤16 的 MLA 部署默认 backend 选择变化,依赖含 flashinfer#2138 的版本;fp8/fp8_ds_mla KV cache 与 32/64 block size 组合进入测试矩阵。

## 关联

前序 sparse backend FLASHMLA_SPARSE(优先级 6);同区域 PR #48770(masked MHA for sparse MLA prefills)、#52212(ROCm Triton sparse-MLA decode);上游依赖 flashinfer-ai/flashinfer#2138。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
