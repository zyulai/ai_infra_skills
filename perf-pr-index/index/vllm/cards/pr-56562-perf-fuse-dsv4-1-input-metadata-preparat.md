# PR #56562: [Perf] Fuse DSV4.1 input metadata preparation with Triton

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-09-12 |
| 分类 | Speculative Decoding |
| Label | performance, ready, deepseek, DSv4 |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含 1 个新文件与 3 个修改文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/56562 |

## 一句话结论

把 DSV4.1 每步都用 PyTorch 算子链搭建的 input metadata(token-to-request 映射、indexer decode 元数据)替换为直接写入既有 buffer 的 Triton kernel,decode 路径元数据耗时降约 4 倍。

## 动机

DSV4.1 在每个模型 step 都要用一长串 PyTorch 操作(如 `torch.repeat_interleave`、`torch.diff`、`copy_`)准备 token-to-request 映射与 flattened indexer decode metadata。Batch-16 decode 单个 builder suite 有 77 次 GPU launch、54 次 `aten::repeat_interleave` CPU 事件,是纯输入准备开销。且 DSpark adaptive verification 在 device 上决定每请求 draft 切分,CPU 侧 boundary 可能是 stale 的,原有基于 CPU 信息的方式不正确也不够快。

## 优化点

新增 `vllm/v1/attention/ops/metadata.py`,包含两个 Triton kernel:`_token_request_mapping_kernel` 在 device 上对 `query_start_loc` 做并行二分搜索,直接为每个 token 生成 request id 并写入既有 buffer(支持 zero-length 请求与 padding,无需中间 tensor);`_indexer_decode_metadata_kernel` 一次 kernel 融合 SM100 varlen indexer 的 sequence-length 展开、block-table 展开、request indices、per-request decode lengths 与 padding 初始化。varlen 路径在 `indexer.py` 的 `build()` 中直接调用该 kernel 填充 5 个既有 buffer,取代原先多个分散的 helper;非 varlen indexer 路径保持不变。所有 kernel 直接调用,不引入新的 JIT warmup 框架封装。

## 关键改动

```text
vllm/v1/attention/ops/metadata.py: 新增两个 Triton kernel(device 二分查找 token->req 映射;融合 indexer decode 元数据构建)
vllm/v1/attention/backend.py: token_to_req_indices() 改用 _token_request_mapping_kernel,基于 device boundary
vllm/v1/attention/backends/mla/indexer.py: 删除 _build_varlen_decode_indices,varlen 路径改调融合 kernel
tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py: 新增元数据正确性测试
```

关键 kernel 逻辑(device 侧二分查找):

```python
lo = tl.full(token.shape, 0, tl.int32)
hi = tl.full(token.shape, num_reqs, tl.int32)
while tl.sum((lo < hi).to(tl.int32), 0) > 0:
    mid = (lo + hi) // 2
    end = tl.load(query_start_loc + mid + 1, mid < num_reqs, other=0x7FFFFFFF)
    right = (lo < hi) & (end <= token)
    lo = tl.where(right, mid + 1, lo)
    hi = tl.where((lo < hi) & ~right, mid, hi)
return tl.minimum(lo, num_reqs - 1)
```

## 性能声明

测试条件:`deepseek-ai/DeepSeek-V4.1-Flash`,4× NVIDIA GB200,TP=4,FP8 KV cache。

- 元数据 microbenchmark(CUPTI GPU 计时,5 个 builder,冷 L2 CUDA-graph replay,batch size 16,仅为 metadata 部分):Decode 191.10→46.27 µs(4.13×),Mixed 220.42→72.48 µs(3.04×),Prefill 168.29→118.21 µs(1.42×),Chunked prefill 1.32×,DSpark verification 3.04×,DSpark adaptive verification 3.33×,DSpark draft 3.15×,DSpark adaptive draft 3.00×。
- Batch-16 decode GPU launches 77→17;`aten::repeat_interleave` CPU 事件 54→0。
- Batch-1 TPOT(512 输入/128 输出):No DSpark 6.091→5.736 ms/token(5.82%),DSpark 1.657→1.407(15.07%),DSpark+adaptive 1.781→1.457(18.20%)。
- SPEED-Bench 8K 输入/1K 输出(DSpark adaptive,`throughput_8k/low_entropy`):TPOT 在 concurrency 1 降 19.31%(3.091→2.494 ms/token),单并发输出吞吐 304.9→373.0 tok/s(1.223×);高并发(8/128/1024)吞吐提升 1.005×–1.032×。

## 验证与回归风险

新增/依赖测试:`test_fused_indexer_decode_metadata`(零长度请求、strided block table、padding、masked 列)、`test_device_token_request_mapping`(CUDA-graph replay 下 CPU boundary stale 时仍跟随 device boundary),以及既有 `test_kpool_tail_slot_mapping.py`、`test_indexer_deepseek_v4_slot_mapping.py`。报告 30 个 attention 测试通过,24 个 metadata-builder case 与 baseline `rtol=atol=0` 完全一致,四 rank 精确 live metadata 对比通过。风险面:仅影响 DSv4.1 varlen indexer 与 `CommonAttentionMetadata.token_to_req_indices` 路径;非 varlen indexer 未改动;GSM8K 1,319 题准确率 96.82%–96.97%(单次运行,非前后等价性测量)。PR 声明为 draft,作者注明仍需人工逐行审查。

## 关联

PR 正文列出的同区域相关工作:#56344(attention math 融合)、#56254(sparse MQA-logit 计算)、#52628(ROCm fused draft-step updates)、#55692(top-k block-table gather 加固)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
