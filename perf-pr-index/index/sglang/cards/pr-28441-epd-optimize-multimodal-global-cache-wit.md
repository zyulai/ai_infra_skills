# PR #28441: [EPD] Optimize multimodal global cache with paged embedding pool

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @cccccya |
| 合并 | 2026-07-07 |
| 分类 | Overlap / Scheduler | tier: major |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 展示 2 个文件(`encode_server.py`、`embedding_cache_controller.py`),另有 3 个文件因超长从 diff 中略去 |
| 链接 | https://github.com/sgl-project/sglang/pull/28441 |

## 一句话结论

多模态全局 embedding 缓存从 flat 连续 CPU buffer 换成分页 host pool,消除碎片导致的分配失败,miss 路径 TTFT 提速 2.0-2.4%。

## 动机

原 multimodal 全局 embedding 缓存把所有 embedding 存在一块 flat CPU pinned buffer 中,用 first-fit 连续分配器管理变长 embedding block。多模态 embedding 的 token 长度各不相同,反复 insert/evict 会把空闲空间切成许多小洞。一旦碎片化,即使总空闲缓存内存仍然很大,新 embedding 也可能因找不到一块连续 CPU 区间而分配失败。eviction 并不能根治:被逐出的条目只与相邻空闲块合并,而被 in-flight RDMA 传输或活跃 CPU view 占用的条目不可逐出。结果是碎片化降低了有效可用缓存容量,导致 cache insert/prefetch 被跳过。

## 优化点

核心是把 embedding 缓存改成分页 host pool:新增 `RangePageAllocator` 与 `PageRun`,一个 embedding 可跨一个或多个物理 page range 存储,缓存元数据从单一 `(offset, size)` CPU block 变为 page run 列表;分配时优先整块连续 page run,碎片化时回退到 scattered page runs。其次按模态拆池:vision pool(image/video)与 audio pool 各自维护独立的分配与逐出状态。第三,引入显式 entry 状态与 pinning:条目标记 `FILLING`/`READY`,读者永远不会消费半写 embedding;Mooncake 传输、host/device copy、pool-view 使用期间条目被 pin,只有 READY 且无活跃 pin 的条目才可逐出。第四,集成 Mooncake multi-buffer scatter/gather API(`batch_get_into_multi_buffers` / `batch_put_from_multi_buffers` 的 wrapper),按每个 embedding 的物理 page runs 构建 pointer/size 列表,分页布局可直接传输而无需先 flatten 成临时连续 buffer。最后重构 encoder 装配路径:`encode_with_global_cache`/`..._mooncake` 拆出 `GlobalCacheEncodeContext` 与 lookup/prefetch/wait/assemble(CPU 与 GPU 两版)/后台 insert 等 helper,Mooncake 路径的 cached embedding 改用 async host-to-device copy,新算出的 miss/fallback embedding 在 async D2H 入池期间保持可用,新存储的 embedding 尽可能后台推送。

## 关键改动

```text
python/sglang/srt/mem_cache/storage/mooncake_store/embedding_cache_controller.py: RangePageAllocator/PageRun/EntryState、按模态分池、Mooncake multi-buffer 集成
python/sglang/srt/disaggregation/encode_server.py: encode 流程重构,分页 embedding 的 CPU/GPU 装配与异步 D2H 入池
```

```python
class RangePageAllocator:
    """Range-aware page allocator that prefers contiguous physical page runs."""
    def allocate(self, num_tokens, page_size):
        ...
        for idx, (start, length) in enumerate(self.free_ranges):
            if length >= required_pages:
                return [PageRun(start, required_pages)]
        # fall back to scattered page runs when fragmented
```

## 性能声明

- Mooncake multi-buffer API 微基准(RoCE v2、`rdma + P2PHANDSHAKE`、segment 32 GB、`MooncakeHostMemAllocator`;multi-buffer 版把 embedding 拆成非连续独立 buffer 以模拟分页池路径):吞吐接近 single-buffer API——512 页 x 128 KB 时开销约 8-10%(如 64MB GET 34.7→31.9 GB/s),页 ≥512 KB 降到 4% 以下,1 MB 页开销接近零(256MB batch4 GET 37.5→37.0 GB/s)。
- 端到端(Qwen3.5-9B,每请求 4 张 2560x1440 图片,3 warmup + 10 measured,临时禁用 image batching 强制走 per-request 全局缓存路径):miss 路径 TTFT,`encode_with_global_cache_mooncake` 3777.8→3687.9 ms(new 快 **+2.4%**),`encode_with_global_cache` 3938.4→3860.8 ms(**+2.0%**);对应 embed 耗时 +0.7%/+3.3%。hit 路径:mooncake hit 基本持平(TTFT -0.7%),**非 Mooncake hit TTFT 回归约 8%**(1761.8→1903.5 ms)。encoder log 确认 hit 场景 `Misses (GPU Work): 0`。
- 池容量/碎片鲁棒性收益:PR 未给出量化数字(设计目标)。

## 验证与回归风险

测试:`test/registered/unit/mem_cache/test_embedding_cache_controller.py` 新增/更新(cache-controller 行为)。PR 声明不改 model forward 计算与输出,只改 embedding 的缓存存储、传输与装配。回归风险:非 Mooncake hit 路径在该次基准中 TTFT 回归约 8%(作者如实报告,未给修复);multi-buffer RDMA 开销依赖页大小,128 KB 页有 8-10% 传输损耗,页大小由 `TARGET_PAGE_BYTES=256KB` 与 embedding 维度推导;FILLING/READY + pin 状态机引入新的失效路径(如 view 未释放会阻止逐出);checklist 各项均未勾选。

## 关联

Mooncake embedding store / L2 缓存与 EPD(encoder-prefill-disagg)架构下的全局缓存路径。无正文列出的具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
