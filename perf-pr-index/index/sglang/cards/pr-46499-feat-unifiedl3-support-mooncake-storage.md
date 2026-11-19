# PR #46499: [Feat] UnifiedL3: support mooncake storage backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Shunkangz |
| 合并 | 2026-11-19 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | documentation, quant, hicache, run-ci, jit-kernel, memory-pool, mooncake, run-ci-nv-extra |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 14 个文件、其中 2 个新测试文件,另有 8 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/46499 |

## 一句话结论

把 HiCache 的统一 L3 key scheme(`--hicache-storage-key-scheme unified`)从 file 后端扩展到 Mooncake Store:对象按网格 chunk(层区间 x kv-head 组,K/V 合一)而非 writer rank 命名,同一模型跨 attention-TP/PP 拓扑共享缓存;32K host-reload 差距收敛(unified 62.3 ms vs legacy 60.7 ms,此前 unified 约 121 ms),page-unified TMA 拷贝比强制 register 路径快 2.87x。

## 动机

此前 unified key scheme 仅支持 file 后端。目标场景:Qwen3-30B-A3B(BF16,TP2,page size 64)on 8x B200,HiCache 由 Mooncake Store 支撑,对比 legacy rank-suffix keys(page_first,Mooncake 默认)与本 PR 的 unified keys(默认分区与 24-layer x 2-head 网格),并保证每次 L3 read 都是 full Mooncake hit。

## 优化点

- 门控统一:新增 `UNIFIED_KEY_SCHEME_STORAGE_BACKENDS = ("file", "mooncake")`,`hicache_hook` 与 `cache_controller` 两处校验共用,替代原来的仅 file 检查。
- 布局与零拷贝元数据:`UnifiedKVPlan.chunk_spans(layout)` 生成每个 suffix 的 chunk 字节 (offset, length) 并校验 head 组数一致;host pool base 新增 `_page_unified_buffer_meta`(每页一段 (address, size));mha/mla 的 `get_page_buffer_meta` 在 page_unified 布局下走零拷贝路径。
- Mooncake 侧:`unified_suffixes`/`unified_chunk_spans` 配置校验(每 suffix 必须有一个 span);`gb_per_page` 按本 rank 拥有的 chunk spans 之和计;`_get_unified_buffer_meta` 以 page-major/chunk-minor 生成 key/ptr/size;`batch_get_v1` 对"Mooncake 把短对象读进大缓冲并报成功"的情况按 size 不符判失败,防 chunk 剩余部分留 stale bytes;`batch_exists` 要求页面命中当且仅当本 rank 拥有的所有 chunk 都在;hybrid pool 在 unified 下显式 NotImplementedError。
- 文档:hicache_design.mdx(PP 各 stage 以自身 suffixes 存对象,unified 下用绝对层区间)与 mooncake README(单 MHA/MLA KV pool;composite caches、speculative decoding、DCP 不支持;page 是 hit 仅当其所有 chunk 都在)。

## 关键改动

```text
python/sglang/srt/arg_groups/hicache_hook.py:
+UNIFIED_KEY_SCHEME_STORAGE_BACKENDS = ("file", "mooncake")
python/sglang/srt/mem_cache/hicache_key_scheme.py:
+    def chunk_spans(self, layout: PageUnifiedLayout) -> list[tuple[int, int]]:
+        ...
+        return [
+            layout.chunk_span(layer_range=layer_range, head_group=head_group)
+            for layer_range in self.layer_ranges
+            for head_group in range(self.head_group_num)
+        ]
python/sglang/srt/mem_cache/pool_host/base.py:
+    def _page_unified_buffer_meta(self, indices): ...
python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:
+    def _get_unified_buffer_meta(self, keys, indices): ...
+        if self.unified_suffixes is not None:
+            get_results = [
+                result if result == size else -1
+                for result, size in zip(get_results, buffer_sizes)
+            ]
test/registered/hicache/test_hicache_storage_mooncake_unified.py(新)、test_hicache_storage_mooncake_unified_cross_tp.py(新)。
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。环境:Qwen3-30B-A3B BF16,TP2,page 64,8x B200,`lmsysorg/sglang:dev`(PyTorch 2.14.1+cu130,Mooncake 0.3.14),Mooncake TCP;2026-11-09 rebase 到 main `e0187268f6`,benchmarked head `e34578eae3`,含已合并的 page-unified TMA kernel #48786。)

- "The 32K host-reload gap is closed: unified takes **62.3 ms** and unified grid **61.4 ms**, versus **60.7 ms** for legacy. The previous unified results were about 121 ms."
- L3-hit TTFT:4K prefix 152.2 / 146.6 / 151.8 ms(legacy/unified/grid);32K prefix 781.5 / 786.9 / 794.2 ms。
- Host-to-GPU load-back bandwidth(server metric):49.3 / 49.2 / 49.2 GB/s;48 并发 8K-token L3 loads:47.0K / 48.1K / 46.9K tok/s;Prefill + Mooncake backup settling(48 x 8K):30.6 / 30.6 / 30.7 s;Multi-turn TTFT avg/p90(256 clients x 6 rounds,16 req/s):42.7/94.6、43.9/95.6、41.8/89.8 ms;multi-turn time per output token 4.77 / 4.71 / 4.60 ms。
- 同构建拷贝基准(48 层,32K tokens,1.5 GiB per rank,十次计时):default unified TMA **31.55 ms / 51.0 GB/s** vs forced register path(quota 2)**90.48 ms / 17.8 GB/s** = **2.87x faster**;legacy `page_first` TMA 32.07 ms / 50.2 GB/s。
- 正文注明:32K L3 会话间波动大(unified 794.8 与 1599.0 ms,grid 1507.7 与 783.1 ms,legacy 787.8 与 776.5 ms,全部计入 pooled medians);"The cause of the L3 variation was not established"。

## 验证与回归风险

跨拓扑(TP2 写 → TP1xPP2 读):GSM8K after L3 reload 60/64(cold 同 60/64),teacher-forced KL 0.0011(限 0.005);TP1xPP2 写 → TP2 读:59/64,KL 0.0010。同拓扑:TP2 下 400 题的 bit-for-bit 重载检查两种 key 均通过;GSM8K legacy 92.25% cold / 91.25% reload,unified 92.25% / 92.50%,unified grid 92.25% / 92.00%。新 E2E:`test_hicache_storage_mooncake_unified.py`(MHA/MLA 的 bit-exact L3 reload,GSM8K 前后差 <0.03)与 `..._cross_tp.py`(TP2↔TP1 缓存迁移,首 token logprob 容差 0.5;H200 实测 TP1 vs TP2 差 ≤0.06,换 chunk 则 2.0-4.5)。`test_page_unified_host_pool.py` 加入 `_check_zero_copy_page_meta`。Rebase validation:166 unit tests 加 60 subtests,32 GPU tests passed。CI 状态:Base 通过,Extra 与 AMD 失败。回归风险:(1) unified 在 Mooncake 下依赖零拷贝 meta 与 chunk span 的一致性(计划/池 head 组数不一致现在在 chunk_spans 内抛错);(2) 短对象/部分 chunk 的失败判定;(3) 仅支持单一 MHA/MLA KV pool,组合缓存、投机解码、DCP 不支持;(4) 32K L3 波动原因未查明。

## 关联

#48786(page-unified TMA kernel,benchmark head 已包含);mooncake_store 目录 README 与 hicache_design.mdx 文档更新。本批次内无直接依赖项。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
