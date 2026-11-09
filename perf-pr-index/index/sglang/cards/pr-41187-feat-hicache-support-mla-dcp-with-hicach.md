# PR #41187: [Feat] hicache: Support MLA DCP with HiCache L3(file/mooncake)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ByronHsu |
| 合并 | 2026-10-23 |
| 分类 | Attention / Kernel |
| Label | documentation, hicache, run-ci, release-highlight, unified-radix-cache, highest-priority, mooncake |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(6 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/41187 |

## 一句话结论

解除 HiCache L3 对 DCP 的拒绝,让 MLA decode 上下文并行的 KV 可落 L3(file/mooncake)复用。

## 动机

此前 `--hicache-storage-backend`(L3)与 `--dcp-size > 1` 组合会在启动时被直接拒绝:DCP 下每个 rank 持有不同的交错 MLA KV shard,rank-0-only 的复制式备份与存储 key 都不是 dcp_rank-aware 的,MLA + DCP 部署因此用不上 L3 层级缓存的跨请求/跨实例前缀复用。

## 优化点

本 PR 是 MLA + DCP + L3 栈的第一块(正文注明:#41187 为 file storage,#41265 为 Mooncake storage,#41591 为 hybrid MLA/KDA storage)。三点核心:其一,允许 file L3 与 DCP 组合,前提是"一个已物化的 MLA host pool",布局支持 layer_first/page_first/page_first_direct;split buffer、独立 scale buffer、额外 pool 仍不支持。其二,把"谁写存储"从 `is_mla_model` 推导改为显式 `is_storage_owner`:每个不同 MLA shard 选一个 writer(如 TP=4/DCP=2 时 rank0 写 shard 0、rank2 读,rank1 写 shard 1、rank3 读,每逻辑页 2 个文件),避免重复写入与逐出记账错误;file evictor 的归属也按此判定。其三,把逻辑页位置翻译成物理行(host pool 中 index // dcp_size),且只统计完整逻辑页——两个 shard 都在才算页命中,任一缺失则停在该页之前;TRAILING_PAGES 命中策略下各 rank 以"每个前缀长度一个 restorable 标志"做 all-reduce,避免 MIN 选中实际缺失的 checkpoint。prefetch 容量预算从物理行改为逻辑 token(mem_pool_host.logical_size)。KV 格式/布局尚未进入存储 key(TODO),共享文件的引擎须用兼容格式与布局。

## 关键改动

```text
python/sglang/srt/mem_cache/hicache_storage.py: HiCacheStorageConfig 增加 dcp_size/dcp_rank/logical_page_size 与 is_storage_writer
python/sglang/srt/managers/cache_controller.py: attach 校验、backup_skip 改为 is_storage_owner、逻辑容量预算
python/sglang/srt/mem_cache/pool_host/mla.py: _storage_page_index 逻辑页到物理行翻译与零拷贝 meta
python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py: restorable 前缀页集合与 TRAILING_PAGES 同步
```

## 性能声明

PR 正文无吞吐/延迟性能数字。价值在于功能解锁:DeepSeek-V2-Lite-Chat(BF16,H200,TP=4/DCP=2,page_first,write-through)手工验证显示 L3 复用成立——cold writer 写入 1025 input tokens 产生 16 个文件、零命中;fresh reader 读取 1024 个 L3 tokens 输出 ID 一致;页 2 缺一个 shard 时仍恢复 128 个 L3 tokens 且输出一致。

## 验证与回归风险

在既有 CI 注册的 file-backend 与 host-pool 测试中新增 6 个聚焦 CPU 测试(不新建文件);CPU 回归共 62 tests 与 40 subtests 通过,覆盖 DCP key/归属、页映射、文件逐出、attachment 与 prefetch;另有上述 H200 手工 e2e(149.48s 通过)。回归面:HiCacheStorageConfig 校验要求 TP 内 DCP 组连续、逻辑页大小被 dcp_size 整除;buffer_only host memory 在 DCP+L3 下仍被拒绝;hit 查询的 all-reduce 载荷形状在 TRAILING_PAGES 路径变化;非 DCP 路径的 MLA 备份从"仅 rank0"变为"每 shard 一个 writer"(dcp=1 时即 rank0,行为等价)。

## 关联

同栈:PR #41265(Mooncake storage)、#41591(hybrid MLA/KDA storage);前序 DCP 工作:#14194(DCP 基础)、#21637(DCP a2a/fi_a2a 通信后端)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
