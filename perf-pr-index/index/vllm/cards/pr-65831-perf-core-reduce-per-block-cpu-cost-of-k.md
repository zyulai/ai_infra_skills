# PR #65831: [Perf][Core] Reduce per-block CPU cost of KV block allocation and caching

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-11-06 |
| 分类 | Quantization \| tier: major |
| Label | ready, kv-connector, scheduler, kv-cache-manager, nixl |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/65831 |

## 一句话结论

降低异步 KV load 每 block 的调度器 CPU 开销，140K request 的 start+promote+finish 降约 50%（2.66→1.34 ms）。

## 动机

调度器主线程上 async KV load 的 start、promote to running、finish 的 per-block Python 开销随 prompt 长度线性增长；在 DP decode 实例上，一个 rank 上偏慢的 schedule() 会拖住每步的 all-reduce。正文说明不打算改变行为，且这些路径被所有请求共享，因此非 P/D 流量同样受益。

## 优化点

block_hash 与 block_hash_num_tokens 变为普通字段，去掉热循环里的 Python property 读；get_new_blocks 对未 hash 的 block 跳过 eviction、_remove_cached_block_hashes 早返回，并按 mode 拆成独立循环，使默认路径免去做 per-block metrics 检查；_insert_block_hash 仅在 block 已有 hash 时才做 contain 检查；cache_full_blocks 每次调用只打包一次 group id；estimate_cached_tokens 改为反向扫描，命中最后一个 hashed block 即停；NIXL pull scheduler 用 len([...]) 代替 sum(genexpr)。源码净变化 +51/-49 行，并移除两个 property。

## 关键改动

```text
vllm/v1/core/block_pool.py: get_new_blocks / cache_full_blocks / hash 插入与删除的热路径优化
vllm/v1/core/kv_cache_utils.py: KVCacheBlock 的 hash 字段由 _block_hash 改为公开普通字段
vllm/v1/core/kv_cache_manager.py: estimate_cached_tokens 改反向扫描
vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py: len([...]) 替代 sum(genexpr)
tests/v1/core/test_prefix_caching.py、test_scheduler.py 等: 回归测试
```

```python
            else:
                for block in ret:
                    # Most blocks were never cached, so skip the call when possible.
                    if block.block_hash is not None:
                        self._maybe_evict_cached_block(block)
                    assert block.ref_cnt == 0
                    block.ref_cnt = 1
```

## 性能声明

正文 Claims：140K-token request 的 start + promote + finish CPU 时间在 DeepSeek-V4.1-shaped KV layout 下降约 50%（2.66 → 1.34 ms），在 one group、block size 16 时下降约 45%（10.55 → 5.79 ms）。分阶段（DSv4.1）：Start fresh 0.71→0.33 ms（-54%）、Promote 1.01→0.65 ms（-36%）、Finish 0.95→0.36 ms（-62%）、Start evicting cached blocks 0.99→0.72 ms（-27%）。另有 block size 64 单组约 -42%（3.04 → 1.75 ms），以及 get_new_blocks loop split 的独立测量 8,750 fresh blocks、caching on、metrics off：2.75 → 1.98 ms（-28%，x86 laptop / Python 3.13）。测试条件：真实 Scheduler + NIXL scheduler connector、无 GPU 与传输、GB200 host（aarch64）、Python 3.12、每轮 32 requests、3 轮各取 5 runs 中位、140K tokens。正文明确标注 "Not yet shown: an end-to-end improvement"，这些是 scheduler-only 数字、无真实 P/D run，对 serving 中 schedule() 时间与 GPU wait 的影响未测；DSv4.1 layout 的 128-token window 与单压缩组是假设，非真实启动日志确认。

## 验证与回归风险

在 GB200 集群跑 test_prefix_caching.py、test_scheduler.py、test_async_scheduler.py、single_type_kv_cache_manager、deferred_block_free、mamba_align_chunk_split、contiguous_kv_packing、simple_kv_offload、kv_connector/unit 等；19 个测试在改动与 baseline 上串行跑都失败（gated models / engine startup / 环境问题），无一方独有；隐藏 GPU 的一次全量跑 4048 passed；新增 test_estimate_cached_tokens_tracks_hashed_prefix_of_each_group 在两边都通过。ruff 通过，mypy 仅在跳过 import 下跑改动文件（弱于仓库 hook）。风险：正文自述测试与表内数字早于最终 get_new_blocks loop split，该拆分尚未过测试、CI 将是首次；kv_cache_metrics 开启时每 block 仍走 eviction call、收益更小且未测；get_new_blocks 现有三个近似相同循环；影响所有请求的 KV block 分配/缓存、prefix caching、P/D 与 NIXL 路径。

## 关联

#61725（限制每步 load start 数以约束该成本）、#64379 与 #65285（针对重复 reservation sum 的独立开销）、#63231（改默认值，正文认为不应影响 NVIDIA 上的 DSv4.1）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
