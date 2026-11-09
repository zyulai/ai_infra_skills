# PR #4353: [Kernel] Use flashinfer for decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LiuXiaoxuanPKU |
| 合并 | 2024-05-03 |
| 分类 | Attention / Kernel(tier: major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 12 个文件(含新增 flashinfer.py) |
| 链接 | https://github.com/vllm-project/vllm/pull/4353 |

## 一句话结论

首次把 FlashInfer 集成进 vLLM:decode 阶段改用其 paged KV decode kernel,prefill 仍走 FlashAttention。

## 动机

2024 年初 vLLM 的注意力后端只有 XFormers/FlashAttn/ROCM/SDPA 几种,decode 阶段缺少针对 paged KV cache 优化的专用 decode kernel。FlashInfer 提供了 `BatchDecodeWithPagedKVCacheWrapper` 等专门面向 decode 的内核,值得作为可选后端接入。与 @yzh119 讨论后确定了边界:关闭 FlashInfer 自带 rope(用 vLLM 的)、alibi slope 与 CUDA graph 支持留待 FlashInfer 后续改进。

## 优化点

新增 `FlashInferBackend` 注意力后端,decode 阶段调用 FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper.forward`,prefill 阶段继续用 `flash_attn_varlen_func`(二者按 `num_prefill_tokens`/`num_decode_tokens` 分派,且不支持 chunked prefill 混合批次)。为匹配 FlashInfer 要求的 NHD 布局 `(num_blocks, 2, block_size, num_heads, head_size)`,在 CUDA 侧新增 `reshape_and_cache_flash` 写缓存 kernel(仅支持 auto dtype,不支持 fp8);decode metadata 构造 `paged_kv_indices/indptr/last_page_len` 三元组并分配 16MB workspace buffer;`pos_encoding_mode="NONE"` 关闭 FlashInfer 的位置编码。选择该后端时强制 eager 模式(尚无 CUDA graph),且不支持 prefix caching、sliding window、alibi;支持 head_dim 64/128/256。TP 场景下 `decode_wrapper` 字段从 zero-copy dict 中跳过以避免 NCCL 广播。

## 关键改动

```text
vllm/attention/backends/flashinfer.py: 新增 FlashInferBackend/FlashInferImpl,decode 走 BatchDecodeWithPagedKVCacheWrapper
csrc/cache_kernels.cu: 新增 reshape_and_cache_flash_kernel(NHD 布局 KV 写入)
vllm/worker/model_runner.py: 为 flashinfer 构造 paged_kv_* metadata 与 16MB workspace buffer
vllm/attention/selector.py: _Backend 枚举新增 FLASHINFER(强制 eager)
```

```python
self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    self.workspace_buffer, "NHD")
...
output = attn_metadata.decode_metadata.decode_wrapper.forward(
    query, kv_cache, sm_scale=self.scale)
```

## 性能声明

PR 正文未提供 benchmark 数字(正文明言这是 "first attempt" 集成,性能对比未附)。

## 验证与回归风险

新增 `tests/kernels/test_cache.py::test_reshape_and_cache_flash`(全参数组合矩阵,fp8 skip);`tests/basic_correctness/test_basic_correctness.py` 在 `VLLM_ATTENTION_BACKEND=FLASHINFER` 下跳过非 eager 用例;`tests/distributed/test_basic_distributed_correctness.py` 验证 TP=2 正确性(强制 eager)。风险点:仅覆盖 GQA/MHA、fp16/bf16 KV cache、head_dim 64/128/256,不支持 fp8 KV cache、prefix caching、chunked prefill、sliding window、alibi 模型,误用面通过 assert/raise 拦截;FlashInfer 依赖为可选 import,未安装时回退为 None。

## 关联

同区域(attention 后端/decode kernel):后续 FlashInfer 集成系列 #11394(flashinfer sampling)、#20034(MLA flashinfer ragged prefill)、#20691(SM100 fused allreduce)、#21078(trtllm-gen decode MLA)等均由此发端。
