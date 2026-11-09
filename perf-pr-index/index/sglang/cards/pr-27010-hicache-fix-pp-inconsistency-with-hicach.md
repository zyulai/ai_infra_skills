# PR #27010: [HiCache] Fix PP inconsistency with HiCache L3 (#22607)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @stepinto |
| 合并 | 2026-08-25 |
| 分类 | KV Cache / Memory |
| Label | high priority, hicache, run-ci, bypass-fastfail, run-ci-extra, release-highlight, unified-radix-cache |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 4 个可见文件(cache_controller.py、scheduler.py、hicache_storage.py、hiradix_cache.py 等),另有 6 个文件从 diff 中裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/27010 |

## 一句话结论

修复 HiCache L3 与 pipeline parallel 并用时多 PP rank 状态发散导致崩溃的问题,引入 pp_sync 事件计数对齐与 prefetch 结果 all_reduce(MIN) 两类同步机制。

## 动机

HiCache(L3 分层缓存)与 PP 同时部署时,多个 PP rank 的缓存状态逐渐发散,最终导致 sglang 崩溃,细节见 issue #22607。根因是各 rank 的 scheduler 对 prefetch/load 完成事件的消费数量不一致:L3 prefetch 的命中长度、完成进度本来是跨 rank 状态,旧实现里 `check_prefetch_progress` 直接在 scheduler 线程做 all_reduce(MIN),而 PP 下这一步无法在 scheduler 线程完成,各 rank 于是对"本次请求实际取回多少 token"得出不同结论。

## 优化点

引入两类同步机制(以 pp_size=2、tp_size=1 举例):

1. **pp_sync(跨 scheduler 线程)**:PP0 的 scheduler 统计自己本地队列(ack_load_queue、ack_prefetch_queue 等)中的完成事件数,PP1 必须从对应队列消费**完全相同数量**的事件,防止发散。实现上把 ack_prefetch_queue 纳入 `_sync_hicache_ready_counts` / `drain_storage_control_queues` 的 all_reduce(MIN) 队列长度对齐,`_drain_queue` 在指定 n 时改为阻塞式 `q.get()`(不足时等待),保证所有 TP/PP rank 消费相同条数。
2. **all_reduce(跨 prefetch 线程)**:PP0 与 PP1 对同一 sequence 的 L3 存储长度可能不同,通过 all_reduce(MIN) 达成一致,与 TP 语义一致。新增 `prefetch_sync_thread` 后台线程消费 `prefetch_sync_queue`,对每条 `PrefetchAck.completed_tokens` 做 all_reduce(MIN) 后放入 `ack_prefetch_queue`;终止判定改为 PP0 决策、其余 rank 跟随(should_terminate all_reduce MAX),并新增 `prefetch_completion_sync_groups`(gloo)与 `prefetch_hits_sync_groups` 分离。KV 与 KV-derived sidecar pool(如 DSA indexer)命中页数取最小 clamping。

## 关键改动

```text
python/sglang/srt/managers/cache_controller.py: 新增 PrefetchAck 数据类、prefetch_sync_thread、prefetch_sync_queue、completion sync groups
python/sglang/srt/mem_cache/hiradix_cache.py: 队列消费按固定条数对齐,check_prefetch_progress 重写为 PP0 决策 + ack 驱动
python/sglang/srt/mem_cache/hicache_storage.py: 抽出 count_pool_hits,extra_pool_hit_pages 改为计数直传
python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py: 同步引入 PrefetchAck(裁剪 diff 中)
python/sglang/srt/managers/scheduler.py: 移除 abort 路径上的 terminate_prefetch 调用
```

关键机制(pp_sync 固定条数消费):

```python
def _drain_queue(q: queue.Queue, n: Optional[int]):
    if n is None:
        while not q.empty():
            item = q.get()
            yield item
    else:
        for _ in range(n):
            # Block when there is no enough elements.
            # All TP/PP ranks must consume the same number of elements.
            item = q.get()
            yield item
```

## 性能声明

PR 正文无修改前后性能对比(分类为正确性修复)。附带的描述性测试:

- SWE-bench Verified(DeepSeek-V4-Flash):Group A 基线(commit 5747ed3b,TP=8,L3)resolved 356/500;Group B 本 PR(PP=4,TP=2,L3)resolved 353/500;差异 7 例归因于容器长时间运行后退出。
- 多轮 HiCache 命中率测试(Qwen3-32B,TP=2,PP=4,128 clients × 10 rounds,request-rate 1):L1 命中率第 4 轮起回落(0.63→0.035),L2 第 7 轮起回落(0.85→0.38),L3 全程维持(第 9 轮 0.899,TTFT 0.29s),说明 L3 在 PP 下正常工作。

## 验证与回归风险

新增 `test_unified_radix_cache_hicache_pp_kl.py`;CI PR Test(Base/Extra)通过,AMD ROCm 7.2 一项失败。风险点:同步路径从 scheduler 线程移到独立后台线程(prefetch_sync_thread、prefetch_io_aux_thread 生命周期管理变化),`_page_transfer` 出错时仍需产出与其他 rank 相同数量的 PrefetchAck(作者自注 "This is so tricky");`can_terminate_prefetch` 大幅简化并移除 TP 内 MAX all_reduce,非 PP 场景的终止时序改变,依赖 ack 队列驱动 `_handle_prefetch_result`。

## 关联

前序:#22607(issue,本 PR 修复对象)。同区域 HiCache:#27770(decode 侧 radix cache)、#30535(mamba IO kernel)、#37778(AMD DSV4 HiCache)、#34012(LRU 淘汰优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
