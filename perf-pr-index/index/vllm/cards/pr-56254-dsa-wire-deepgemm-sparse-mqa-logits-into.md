# PR #56254: [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JaredforReal |
| 合并 | 2026-09-15 |
| 分类 | Attention / Kernel |
| Label | performance, ready, deepseek, DSv4, kv-cache-manager, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供);diff 可用但为截断版(10 个文件被省略),保留部分涉及 `sparse_mqa_logits.py`(新)、`config/attention.py` 与大量测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/56254 |

## 一句话结论

V4.1 两级索引的 consumer 层改用 DeepGEMM sparse MQA logits 只对候选块打分,长上下文 prefill 吞吐最高 2x(1M)、内核级最高 22.9x。

## 动机

DeepSeek V4.1 的 token 选择是两级:首个 indexer 层发布上下文的候选块,后续 consumer 层在其中选 top-k。此前 consumer 层仍对**整个上下文**算 dense MQA logits 再 mask 掉候选之外的部分,O(context) 的 logits 计算(且 fp32 logits 要多次分块写读)成为长上下文瓶颈。DeepGEMM 2.8(deepseek-ai/DeepGEMM#432)提供 sparse MQA-logits 内核,只对给定块列表打分,把 logits 开销降到 O(候选块)(V4.1-Flash 固定 16K 位置)。

## 优化点

设计要点:(1) 新建独立 `SparseMQAIndexer` 层 + `DeepseekV41SparseIndexerBackend`/`Metadata`,普通 module 而非 custom op(V4.1 不走 torch.compile);dense `SparseAttnIndexer` 不动,候选源层与其前层保持 dense;(2) 引擎启动时一次性校验全部前置条件(模型、MXFP4 indexer cache、GPU 代际、DeepGEMM>=2.8、DeepSelect 扩展、logits/top-k 布局),失败显式报错而非逐步静默回退,路径上无 PyTorch fallback;(3) 热路径零分配零重算:builder 持有 scratch,候选展开与 DeepGEMM schedule 只依赖候选(同 step 所有 consumer 层相同),首层计算、其余经共享 metadata 复用;(4) 热路径无 dtype cast:fused Q RoPE-quant kernel 为该 indexer 输出 bf16 权重(dense 为 fp32),DeepSelect(#56464)直接在 bf16 logits 上做 top-k,再由 Triton kernel 重映射回上下文位置;(5) Opt-in:`--attention-config '{"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": true}'`,约 16K 以下候选覆盖全上下文、dense 更便宜,默认关;(6) DeepGEMM paged 内核要求页 stride 512B 对齐,`MLAAttentionSpec` 新增 `block_stride_alignment`,block size 就地向上取整(开启时 KV 显存 +0.8%)。

## 关键改动

```text
vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py: 新增稀疏 logits 流水线(候选展开/score/top-k 重映射)
vllm/config/attention.py: 新增 indexer_sparse_logits 开关
vllm/models/deepseek_v41/...: SparseMQAIndexer 层与 backend/metadata(在截断部分)
tests/model_executor/layers/test_mla_short_prefill_indexer.py 等: 展开/选择一致性/数值测试
```

```python
+    indexer_sparse_logits: bool = False
+    """DeepSeek V4.1 two-level indexer: score only the candidate blocks with
+    DeepGEMM's sparse MQA-logits kernels instead of computing dense logits over
+    the whole context and masking them. ..."""
```

## 性能声明

内核级(GB300,sparse 相对 dense indexer 流水线每 consumer 层加速;decode 为 batch 64/256,prefill 为 chunk):

- 8K:1.2x;32K:1.7-1.8x(prefill 2.3x);128K:4.4-5.8x(prefill 8.3x);512K:14.4-22.9x。

端到端(DeepSeek-V4.1-Flash,TP=4 4x GB300,MXFP4 indexer cache,greedy,前缀缓存关,线性拟合吞吐):

- decode:8K-1M 均为 1.03-1.06x(原文 "Decode gets 3-6% faster");prefill:100K 1.06x、256K 1.19x、512K 1.43x、1M **2.00x**;8K prefill 0.95x(慢 5%)。

精度(78K 真实文档按位置分桶 NLL,on/off 差异与两次运行间噪声同级,均 ≤0.002 量级):sparse 内核以 bf16 打分(dense 为 fp32),greedy 输出与 dense 路径不逐 token 一致。KV 显存代价 0.8%(对齐开销)。

## 验证与回归风险

28 个单测通过:候选展开 Triton kernel 对 PyTorch 参考(含 strided 输入、垃圾值/int32 溢出);sparse 选择与 dense bf16 logits mask 后 `torch.topk` 逐值一致(仅并列顺序可不同),覆盖 prefill、paged decode(含 MTP 行)、CUDA-graph capture/replay 与 schedule 复用;fused Q kernel 的 bf16 输出等于 fp32 结果单次舍入。限制:仅 SM100、DeepGEMM>=2.8、需 DeepSelect(`vllm._deepselect_C`)与 MXFP4 indexer cache,不支持 DCP/PCP;`--sparse-indexer-topk-backend` 须为 auto/deep_select。风险面:打分精度从 fp32 降为 bf16(长文本 NLL 影响已测但限于单文档);16K 以下上下文 prefill 变慢。

## 关联

依赖 DeepGEMM 2.8 sparse MQA 内核(#432)与 DeepSelect top-k(#56464)。同批 DeepSeek V4.x 相关:#51967(DSV4 top-k 内核)、#64101(DSv4.1 ROCm mono)、#35891(V3.2 sparse MLA fp8 KV)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
