# PR #27285: [HiCache] Fix crash when using PP + HiCache L2

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27285](https://github.com/sgl-project/sglang/pull/27285) |
| 作者 | stepinto |
| 合并日期 | 2026-06-06 |
| labels | high priority, hicache, run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 3 · 💬 15 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
PP + HiCache L2 下 PP0/PP1 scheduler 线程缺乏同步导致服务崩溃;引入 pp_sync 机制——PP0 统计本地 ack 队列完成事件数并沿流水线传递,PP1 消费完全相同数量的事件。

## 问题现象
HiCache 与 PP(流水线并行)组合使用会导致服务崩溃。正文给出最小复现:`sglang serve --model-path=Qwen/Qwen3-32B --pp-size=2 --enable-hierarchical-cache --max-total-tokens=$((256*1024))` 后跑 `python -m sglang.bench_serving --num-prompts 1000`,报错 `RuntimeError: shape '[3013, -1, 128]' is invalid for input of size 8192000`。作者还用 `bench_multiturn.py`(128 客户端 × 10 轮,pp-size=4/tp-size=2,L1 256K tokens、L2 512K tokens)验证修复后 L2 生效:无 HiCache 的对照组命中率第 4 轮后从 0.34 骤降到 0.09,而 HiCache L2 组在第 4~7 轮保持 0.79/0.83/0.85/0.71 的高命中。本补丁只修 PP + L2,L3 尚未解决。

## 根因
正文披露:崩溃原因是 PP0 与 PP1 的 scheduler 线程之间缺乏同步,细节在 #22607 中讨论。

## 修复方案
为 scheduler 线程引入名为 **pp_sync** 的新同步机制。正文描述:PP0 的 scheduler 线程确定其本地队列(ack_load_queue、ack_prefetch_queue 等)中完成事件的数量,PP1 必须从自己的队列消费完全相同数量的事件,从而防止两边状态分叉。具体改动(由 diff 描述):

- `python/sglang/srt/managers/cache_controller.py`:构造参数废弃 `pp_rank/pp_size`,改为内部通过 `get_pipeline_model_parallel_rank()/get_pipeline_model_parallel_world_size()` 获取,并新增 `pp_group` 参数。
- `python/sglang/srt/mem_cache/hiradix_cache.py` 与 `unified_radix_cache.py`:新增 `_pp_sync(data)`(PPn(n>0) 从 PPn-1 recv 数据,再 isend 给 PPn+1,无需全局同步点,异步 send 用 `work_list` + `_reap_completed_async_work()` 回收)和 `_all_reduce()`(先在 PP0 的所有 TP rank 上做 reduce,结果再沿 PP 传播);`writing_check`/`loading_check` 改为仅 `pp_rank == 0` 统计 `ack_write_queue`/`ack_load_queue` 完成事件数,再经 `_all_reduce(MIN)` 广播给下游 PP rank,各 rank 按同一 finish_count 消费队列:

```python
+        if self.pp_rank == 0:
+            for _, finish_event, ack_list in cc.ack_write_queue:
+                if not finish_event.query():
+                    break
+                finish_count += 1
+        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
+        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
```

- `CacheInitParams`/`kv_cache_builder.py`/`hybrid_pool_assembler.py` 等全链路透传 `pp_cache_group`;`unified_radix_cache.py` 的 `load_back_threshold` 由 256 改为 10。
- 新增测试 `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_hicache_pp_kl.py`(Qwen3-30B-A3B-FP8,TP2+PP2+HiCache+UnifiedRadixCache,GSM8K/MMLU/decode cached_tokens 断言)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响所有启用 HiCache 时 `_all_reduce`/`writing_check`/`loading_check` 的路径——非 PP(pp_size<=1)时 `_pp_sync` 直接返回,行为不变;PP 下新增 rank 间 recv/isend 同步,每轮检查多一次 PP 链上通信,且 PP1+ 的 ack 消费数量改由 PP0 决定,若 PP0 统计偏少会让下游积压;`load_back_threshold` 从 256 降到 10 会更频繁触发 load-back 检查。L3(prefetch/storage)与 `UnifiedRadixCache`/`HybridCacheController` 的完整适配(原 #27010 的范围)仍待后续补丁。

## 关联
- [#22607](https://github.com/sgl-project/sglang/issues/22607)(PP0/PP1 scheduler 线程同步问题细节)
- [#27010](https://github.com/sgl-project/sglang/pull/27010)(被拆分的原大补丁:PP + HiCache L2+L3)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
