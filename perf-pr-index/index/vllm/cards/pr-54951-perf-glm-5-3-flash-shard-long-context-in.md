# PR #54951: [Perf][GLM-5.3-Flash] Shard long-context indexer prefill rows across TP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zigzagcai |
| 合并 | 2026-10-11 |
| 分类 | Speculative Decoding |
| Label | new-model, rocm, speculative-decoding, ready, torch.compile, ci/build, multi-modality, deepseek, kv-connector, nvidia, quantization, mrv2, verified, scheduler, kv-cache-manager, glm, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用,正文注明 diff 共 4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/54951 |

## 一句话结论

把 DSA indexer prefill 的 MQA 打分行按成本均衡分片到各 TP rank,GLM-5.3-Flash 1M 上下文 TTFT -25.42%。

## 动机

GLM-5.3-Flash 这类 DSA(sparse attention)模型在 TP 下,sparse-indexer 的 prefill MQA 打分与 top-k 在每个 TP rank 上对同一批 query row 重复计算(行是复制的)。indexer 的 MQA scoring 工作量随 query 行数与可用 pooled-key 历史长度增长,而按行交换最终 int32 top-k 索引的通信宽度是固定的——上下文越长,可省的冗余计算相对通信开销越大。这是纯冗余消除,不近似注意力、不合并独立 top-k 候选。

## 优化点

核心是 prefill 行级分片:调度侧 `balanced_prefill_row_shard` 按 indexer 的因果工作量(累计 `(first_key + row_in_request) // compress_ratio`)用 searchsorted 找出成本均衡的连续行区间,每 rank 至少 1 行;门控 `tp_prefill_row_sharding_supported` 要求 CUDA、TP>1、无 DCP/PCP、未禁用 PyNCCL、非 NCCL symm-mem/batch-invariant、CUDA-graph 混合模式非 FULL,且当前 batch 至少 `16,384 × TP` 个 prefill query 行(TP4 即 64K 行),否则整体走原复制路径。执行时每 rank 通过 `PrefillRowShard.narrow` 把 chunk 窗口收窄到自己的行区间,只对全局 key 历史打自己行的分;每层 indexer 结束后一次 `all_gatherv` 把各 rank 的最终 int32 索引按全局行序拼回(测试注明这是 API 调用而非单次物理 NCCL 操作)。GLM 的 k-pool 路径交换宽度为 `index_topk + index_kpool - 1`(2048+4-1=2051),把不完整尾部一并交换;KV gather 保持无条件执行以复用 workspace;profiling/dummy run 提前预留交换瞬态内存;decode 内核、权重、采样、KV 格式均不变。

## 关键改动

```text
vllm/v1/attention/backends/mla/indexer.py: balanced_prefill_row_shard 成本均衡切分 + tp_prefill_row_sharding_supported 门控 + row_shard_sizes 元数据
vllm/model_executor/layers/sparse_attn_indexer.py: PrefillRowShard(narrow/exchange_topk)、通用 DSA 路径行收窄与 all_gatherv
vllm/models/glm5next/nvidia/sparse_indexer.py: GLM k-pool indexer 接入分片(交换宽度含尾部)
tests/v1/attention/test_indexer_tp_row_shard.py: 分片覆盖、交换重组、门控拒绝、瞬态预留测试
```

关键片段(行交换):

```python
def exchange_topk(self, topk_buffer: torch.Tensor, width: int) -> None:
    local = topk_buffer[self.start : self.stop, :width].contiguous()
    end = self.num_decode_tokens + sum(self.sizes)
    topk_buffer[self.num_decode_tokens : end, :width] = \
        get_tp_group().all_gatherv(local, dim=0, sizes=self.sizes)
```

## 性能声明

GLM-5.3-Flash,4x H200 TP4,FP8 权重/BF16 KV,C1、256 output tokens、max_num_batched_tokens=65,536、legacy runner + 显式 FLASH_ATTN_MLA_SPARSE 后端,baseline(main 父提交 5893426b88f7)→ PR head(16934c6e76e5),8 个 fresh 进程 ABBA/BAAB 四对、几何平均:

- 128K:TTFT 4.782 → 4.605 s(+3.69%),total tok/s 20269.73 → 20835.80(+2.79%)
- 256K:TTFT 10.154 → 9.372 s(+7.70%),吞吐 +7.07%
- 512K:TTFT 22.760 → 19.418 s(+14.68%),吞吐 +15.78%
- 1M(1,000,000 tokens):TTFT 54.154 → 40.388 s(+25.42%),吞吐 17856.28 → 23675.18 tok/s(+32.59%)
- TPOT 全长度点估计 -0.03% ~ +0.01%,配对 95% 区间均含 0;2K 回退路径 TTFT 点估计 -2.84%(区间 [-8.18, +2.24]% 跨零),作者明确不声称短输入无回退。TP4 + max_num_batched_tokens=8192 无法激活(门控按 batch 行数而非请求上下文长度)。

## 验证与回归风险

测试:新增 `tests/v1/attention/test_indexer_tp_row_shard.py`(分片恰好覆盖一次、交换只落真实 prefill 行、均衡切分与拒绝条件、门控全否决组合、profiling 瞬态预留),54 项 focused tests 全过。正确性:33 个 1K-1M 检索 probe x 4 进程,答案在场 132/132 → 132/132,strict answer-only 127/132 → 129/132,精确 token 一致 130/132 A/B(基线 A/A 98/99),两处差异均为格式(基线多一句解释);另用 DeepGEMM + CUDA top-k 与 Triton pool expansion 做数值等价抽查。风险:结果仅限定 legacy-runner + FA3 配置(V2 runner 在 256K 的 slot-mapping kernel 失败、FlashInfer SM90 sparse MLA 在 65536 行 autotune 失败,均为基线即有);通用 DSA 路径同样被修改,其他模型/平台/并发/MTP 未验证;TPOT 等价未被证明。

## 关联

前序:#53906(已在 baseline,本 PR 不重引其模型支持与内核改动);同区域:#62170/#62171/#62172(cooperative_topk BF16,同一 DSA indexer 家族);模型标签含 DSv4.1(candidate-block 行保持复制)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
