# PR #52162: [Perf][PCP] Shard decode requests across PCP ranks

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pisceskkk |
| 合并 | 2026-10-02 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | ready, deepseek, nvidia, mrv2 |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,13 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/52162 |

## 一句话结论

PCP-only 部署(DCP=1)下把 decode 请求按轮转分片到各 PCP rank,消除 pcp_size 倍重复 decode 计算:128k 上下文 PCP4 吞吐最高 +6.17%。

## 动机

PCP(prefill context parallelism)目前在每个 PCP rank 上复制全部 decode 请求。PCP-only 部署(`DCP == 1`)中每个 rank 本就持有完整 KV cache 副本,同一 decode 计算被重复执行 `pcp_size` 次,纯浪费;长上下文 decode 阶段尤其明显。但分片后为保持各 rank 的 KV 副本一致,需要额外通信传播 owner 产生的 KV 更新——compute-communication 权衡使实际收益小于理想的冗余消除,且依赖 workload。

## 优化点

(1) `PCPManager._iter_rank_chunks` 对 decode-only 行(非 prefill、非 DCP 复制)按每步的紧凑 decode 序号 round-robin 指派 owner rank:owner 执行该请求,其他 rank 留零 token 占位行以保持集合通信的均匀 shape;KV 与 hidden states 会被 gather 回所有 rank,故 ownership 无需跨步持久;(2) `_gather_prefill_cache_inputs` 在 shard 模式下把 owner 产生的 MLA latent KV 与 sparse-indexer K 更新 all-gather 到全部 PCP rank,保证缓存副本完整;DCP>1 时保持 decode 复制不变(每个 DCP rank 须消费本地 KV 分片),DualChunkSwap prefill 分区不变;(3) `_build_batch_layout`:复制行的 KV 写 mask 只保留 rank 0 为规范写入者,sharded decode 恰有一个 owner、无需去重;(4) 边界处理:indexer/FlashMLA sparse FP8 metadata builder 遇到零 token 占位行直接返回空 metadata,不得当成零长 decode 请求;(5) `ParallelConfig.pcp_shard_decode_requests` 属性(PCP>1 且 DCP==1)作为总开关,贯穿 mla_attention/sparse_attn_indexer/deepseek_v32 attention。

## 关键改动

```text
vllm/v1/worker/gpu/pcp_manager.py: PCPManager 增加 shard_decode_requests;decode 轮转分片、KV 写 mask、layout/gather 索引
vllm/v1/attention/ops/pcp.py: _gather_prefill_cache_inputs 在 shard 模式 gather owner 的 decode KV 更新
vllm/config/parallel.py: pcp_shard_decode_requests 属性(PCP>1 且 DCP==1)
vllm/v1/attention/backends/mla/indexer.py, flashmla_sparse.py: 零 token 占位行返回空 metadata
vllm/model_executor/layers/sparse_attn_indexer.py, mla_attention.py, models/deepseek_v32/attention.py: 开关透传
```

关键代码(轮转分片):

```python
elif self.shard_decode_requests:
    chunk_size = query_len
    # decode ownership does not need to persist across steps. Use a
    # compact decode-only ordinal to keep each step exactly
    # balanced even when prefills and zero-token rows are present.
    owner_rank = decode_ordinal % self.pcp_world_size
    decode_ordinal += 1
    chunk_indices = (0,) if rank == owner_rank else ()
```

## 性能声明

NVIDIA H20,128k 输入/1k 输出、并发 32、128 请求、温度 0、seed 0、max_num_batched_tokens=32768,全部 128/128 完成 0 失败。DeepSeek-V3.2 FP8 随机两层 64 专家 checkpoint(拓扑/性能探针,非生产模型估计),按 prefix-cache 命中率分档(PCP 分片前→后):4x H20 PCP4+EP4 在 90/95/97/99% 档 req/s +1.98%/+6.17%/+3.68%/+2.37%(95% 档 2.8343→3.0094 req/s、2,902.4→3,081.6 output tok/s、TPOT 8.84→8.39ms);80% 档复测为 -2.16%。8x H20 PCP8+EP8:95% 档 +3.73%(0.4833→0.5014 req/s,TPOT 58.87→57.08ms)、97% 档 +1.33%,99% 单次测量 -0.86%。正确性:DeepSeek-V2-Lite-Chat 全量 GSM8K 5-shot(1,319 题,温度 0,并发 256),TP2+EP2 64.7460% vs PCP2+EP2 分片 64.6702%,均在噪声内。

## 验证与回归风险

`tests/v1/worker/test_gpu_pcp_manager.py` 新增轮转均衡(含乱序重排后 owner 变化)、DCP 下保持复制、空 owner rank 允许、prefill 分区不变、owner KV 选择等用例;`tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` 增加零 token PCP rank 参与 compressed mapping gather 的用例;cudagraph manager/sparse MLA/short prefill indexer 测试同步更新签名。风险面:改动 PCP 路径的 batch layout、gather 索引与 cudagraph padding 逻辑,影响所有 PCP 用户(DCP>1 行为保持);indexer 与 FlashMLA sparse FP8 metadata 的零 token 早退是新的边界路径;收益依赖 workload(高命中/长上下文场景更明显)。

## 关联

同区域:#49564(FlashAttention/GQA PCP 与 PCP+DCP prefill)、#49517(KV 更新 collective 改 opt-in 直写)、#49741(owner-shard 持久 KV history)、#31473(PCP 性能 tracking);DSA 路由背景见 PR #52861。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
