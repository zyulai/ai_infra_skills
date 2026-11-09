# PR #8133: [Performance][PD Disaggregation] optimize TokenToKVPoolAllocator by sorting free pages

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YiXR |
| 合并 | 2025-07-23 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,仅涉及 python/sglang/srt/mem_cache/allocator.py |
| 链接 | https://github.com/sgl-project/sglang/pull/8133 |

## 一句话结论

PD 分离场景下延迟释放并排序合并 KV cache 空闲页,减少碎片、提升 p2p 传输效率,TTFT 最高降 25%。

## 动机

大规模分布式推理(尤其 PD Disaggregation)中,KV cache token 索引的高效管理至关重要。空闲索引碎片化会使 peer-to-peer 传输难以分配大块连续区域,导致通信模式低效,直接推高端到端延迟尤其是 TTFT。此前的 allocator 在 free 时立即把索引放回 free pool,高并发下碎片与分配争用明显。

## 优化点

核心是引入"延迟释放 + 按需合并排序"的页管理机制:

1. **release_pages 缓冲**:新增 `release_pages` 张量,`free()` 释放的页索引不再立即拼入 `free_pages`,而是先进入缓冲区;`available_size()` 返回两者之和,对上层保持容量语义不变。
2. **按需 merge_and_sort_free()**:当分配请求无法由当前 `free_pages` 满足时(alloc / alloc_extend / alloc_decode 三个入口均按估算新增页数判断),才把缓冲页与 free pool 合并并 `torch.sort`,使相邻空闲区域合并成大块,缓解碎片化。
3. **状态快照适配**:`backup_state()`/`restore_state()` 改为同时保存 `(free_pages, release_pages)` 二元组,`clear()` 时重置缓冲区。

排序触发是惰性的,只在空闲页不足时付出一次 sort 成本,正常分配路径无额外开销。

## 关键改动

```text
python/sglang/srt/mem_cache/allocator.py: 唯一改动文件。BaseTokenToKVPoolAllocator
与 PagedTokenToKVPoolAllocator 增加 release_pages 缓冲、merge_and_sort_free();
free() 改写入缓冲;alloc/alloc_extend/alloc_decode 在估计页数超过 free_pages 时
触发合并排序;backup/restore state、clear、available_size 同步适配。
```

关键片段(diff 摘录):

```python
def merge_and_sort_free(self):
    if len(self.release_pages) > 0:
        self.free_pages = torch.cat((self.free_pages, self.release_pages))
        self.free_pages, _ = torch.sort(self.free_pages)
        self.release_pages = torch.empty(
            (0,), dtype=self.release_pages.dtype, device=self.device
        )
```

## 性能声明

PR 正文测试条件:2 × A10 × 8 节点(RDMA),Qwen 0.6B,1P1D 部署,bench 结果 400 请求、16 并发:

| 场景 | 优化前 TTFT | 优化后 TTFT | 提升 |
| --- | --- | --- | --- |
| 1024 in / 512 out | 81.29 ms | 69.35 ms | 14.8% |
| 2048 in / 1024 out | 144.02 ms | 108.04 ms | 25% |

附测数据显示 Total token throughput 3264.53→3320.14 tok/s(1024/512)与 3103.98→3132.99 tok/s(2048/1024);PR 另给出 send_kvcache() 需传输 kv_blocks 数量对比图(1024 in/512 out),显示传输块数减少。ITL 基本持平。

## 验证与回归风险

- PR 未添加单元测试(Checklist 中 "Add unit tests" 未勾选),仅提供 benchmark 数据。
- `backup_state()`/`restore_state()` 接口从单张量变为 tuple,依赖 allocator 状态快照的路径(如 overlap scheduler、radix cache、PD 分离状态保存)需同步适配,否则状态恢复不完整。
- `available_size()` 语义改为 free + release 之和,上层据此做容量判断,行为一致性依赖 merge 触发条件正确。
- 惰性 sort 只在页不足时触发,但 alloc_extend/alloc_decode 中用 `.item()` 同步估算页数,存在 host-device 同步点,对调度热路径有轻微影响。
- 影响 TokenToKVPoolAllocator / PagedTokenToKVPoolAllocator 两族分配器(int64 与 int32 两种页索引 dtype 分支均有改动)。

## 关联

无(pack 素材未提供同区域其他 PR 信息)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
