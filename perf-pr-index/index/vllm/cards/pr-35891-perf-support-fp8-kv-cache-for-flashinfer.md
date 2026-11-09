# PR #35891: [Perf] Support FP8 KV cache for Flashinfer MLA Sparse

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-03-07 |
| 分类 | Quantization |
| Label | documentation, ready, v1, nvidia |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 sparse MLA 两个后端、mla_attention.py、模型 config、测试与文档生成脚本 |
| 链接 | https://github.com/vllm-project/vllm/pull/35891 |

## 一句话结论

让 FLASHINFER_MLA_SPARSE 后端支持 fp8 KV cache(配合 fp8 query 量化),DeepSeek-V3.2-NVFP4 吞吐比 FlashMLA sparse 高 14%。

## 动机

DeepSeek V3.2 的 sparse MLA 注意力有两个后端:FLASHMLA_SPARSE(仅支持 DeepSeek 自定义 `fp8_ds_mla` KV 格式)与 FLASHINFER_MLA_SPARSE(TRTLLM-GEN sparse MLA 内核,此前只支持 auto/bfloat16)。用户开 `--kv-cache-dtype fp8` 时会被强制转到 `fp8_ds_mla`(模型 config 里写死),无法用上 FlashInfer 后端的 fp8 路径。本 PR(tracking #35805)打通 FlashInfer MLA Sparse 的标准 fp8 KV cache 支持。

## 优化点

(1) `FlashInferMLASparseBackend.supported_kv_cache_dtypes` 加入 `fp8`、`fp8_e4m3`,并声明 `supports_quant_query_input = True`——TRTLLM-GEN sparse MLA 内核要求 query 与 KV cache dtype 一致,不支持 bf16 query + fp8 KV 混合;(2) `mla_attention.py` 的稀疏 MLA 公共路径在 fp8 attention 时把 `kv_cache` view 成 fp8 dtype,并在后端支持量化 query 输入时用 `_DecodeConcatQuantFP8`(static、PER_TENSOR、compile_native)把 nope/pe 两段 query 拼接量化后传入 `forward_mqa`,否则维持 tuple 传入;(3) FLASHMLA_SPARSE 收到非 `fp8_ds_mla` 的 fp8 请求时自动转成 `fp8_ds_mla` 并提示;FLASHINFER_MLA_SPARSE 则保留标准 fp8 并反向提示;(4) 移除 DeepseekV32 config 中强制的 fp8 → `fp8_ds_mla` 改写,把选择权交回后端;`fp8` 成为 FlashMLA Sparse 的别名。

## 关键改动

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py: 支持 fp8/fp8_e4m3 KV,声明量化 query 输入
vllm/model_executor/layers/attention/mla_attention.py: fp8 时 KV view + query 拼接量化;两后端 dtype 归一
vllm/model_executor/models/config.py: 去掉 V3.2 强制 fp8_ds_mla 改写
tests/v1/attention/test_sparse_mla_backends.py 等: fp8 正确性测试与跳过规则
docs/design/attention_backends.md + tools/pre_commit: 后端支持矩阵更新
```

```python
+        if fp8_attention and kv_cache_dtype != "fp8_ds_mla":
+            kv_cache = kv_cache.view(current_platform.fp8_dtype())
...
+        if fp8_attention and self.impl.supports_quant_query_input:
+            mqa_q = self._decode_concat_quant_fp8_op(
+                mqa_ql_nope, mqa_q_pe, self._q_scale
+            )
+        else:
+            mqa_q = (mqa_ql_nope, mqa_q_pe)
```

## 性能声明

nvidia/DeepSeek-V3.2-NVFP4,`--kv-cache-dtype fp8`,DP8/TP1、max-num-seqs 1024、random input 1024 / output 1024、1280 prompts、concurrency 256(硬件型号 PR 未注明,仅给出 `--gpu-memory-utilization 0.8` 等配置):

- FLASHMLA_SPARSE(fp8_ds_mla):Output token throughput 6868.29 tok/s,Mean TTFT 3203.28 ms,Mean TPOT 34.12 ms
- FLASHINFER_MLA_SPARSE(fp8):Output token throughput 7810.91 tok/s,Mean TTFT 2287.03 ms,Mean TPOT 30.52 ms

即 FlashInfer 后端相对 FlashMLA 吞吐 **14% speedup**(7810.91/6868.29),TTFT/TPOT 均更低。

## 验证与回归风险

测试计划:`tests/v1/attention/test_sparse_mla_backends.py` 新增 FlashInfer fp8 单测;decode correctness 测试改为直接用配置的 kv_cache_dtype,并显式跳过 FlashMLA Sparse 的非 `fp8_ds_mla` fp8 用例(该后端只支持 fp8_ds_mla);文档生成脚本为 FLASHMLA_SPARSE 排除 `fp8` 别名以避免矩阵歧义。风险面:fp8 query 量化路径改变传入内核的 dtype(精度影响 PR 未给出数值对比,仅靠正确性单测);DeepSeekV32 的 KV 格式选择从 config 层移到 backend 层,依赖 `supports_quant_query_input` 这类后端能力位;scale 处理沿用既有 `_q_scale`/`_k_scale`(静态)。

## 关联

同 DeepSeek V3.2 sparse attention 区域:#56254(DSA:DeepGEMM sparse MQA logits 接入 V4.1 indexer)。跟踪 issue #35805。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
