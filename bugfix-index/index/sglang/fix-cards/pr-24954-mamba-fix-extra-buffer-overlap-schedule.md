# PR #24954: [Mamba] Fix extra_buffer overlap schedule races

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24954](https://github.com/sgl-project/sglang/pull/24954) |
| 作者 | hanming-lu |
| 合并日期 | 2026-05-19 |
| labels | npu, run-ci, run-ci-extra |
| 热度 | 👍 6 · 💬 24 |
| 分类 | Overlap / Scheduler |
| bug 类型 | general |

## 一句话结论
Mamba 状态操作与 forward 流存在数据竞态,把 COW/清零/缓存捐献全部移到 forward 流上执行以消除竞态。

## 问题现象
开启 mamba extra buffer 的 overlap 调度下,scheduler 流与 forward 流并发操作 mamba 状态产生数据竞态。正文未描述具体报错日志;作者在 issue #24221 的讨论中说明该修复经 0kl RL 测试验证为无竞态(race-free)。

## 根因
PR 正文披露:此前 mamba 状态的读写发生在 scheduler 流上,与 forward 流上的计算并发进行,存在 `scheduler <-> forward stream data manipulation races`,具体两处:
1. `cache_unfinished_req` 在缓存未完成请求时对 mamba state 做拷贝,该拷贝与 forward 流竞态;
2. prefix match 的 copy-on-write(COW)与 mamba state 清零发生在 scheduler 流上。

## 修复方案
目标:把所有 mamba 状态操作移到 forward 流。主要改动:
- `schedule_batch.py`:`Req` 新增 `mamba_cow_src_index`、`mamba_needs_clear` 延迟操作字段;`ScheduleBatch`/`ForwardBatch` 携带 `mamba_cow_src_indices`、`mamba_cow_dst_indices`、`mamba_clear_indices`;`prepare_for_extend` 调用 `_collect_deferred_mamba_cow_and_clear` 收集延迟 COW/清零信息;
- `hybrid_linear_attn_backend.py`:新增 `_execute_deferred_mamba_cow_and_clear`,在 `init_forward_metadata`(extend 且非 draft worker 时)于 forward 流上执行 `mamba_pool.clear_slots` 与 `copy_from`,执行后清空标记;
- `mamba_radix_cache.py` / `hi_mamba_radix_cache.py` / `mamba_component.py`:`_match_post_processor`、`finalize_match_result` 不再当场 `copy_from`,改为记录 `req.mamba_cow_src_index`(延迟到 forward 流);`cache_unfinished_req`/`prepare_for_caching_req` 改为「把 mamba index 捐献(donate)给 radix cache 再为下一轮分配新槽」,新增 `_alloc_mamba_slot` 辅助函数;
- `memory_pool.py`:`alloc` 不再在分配时清零,拆出 `clear_slots`(必须在 forward 流执行);删除 `fork_from`;ping-pong track buffer 映射 dtype 由 int32 改 int64;
- `eagle_info_v2.py`、`schedule_batch.py`:统一用新的 `set_mamba_track_indices_from_reqs` 从 req 对象构建 track indices。

关键 diff 摘录:

```python
def _execute_deferred_mamba_cow_and_clear(self, forward_batch: ForwardBatch):
    """Run deferred clear/COW ops on the forward stream to avoid races."""
    if not forward_batch.forward_mode.is_extend() or self.is_draft_worker:
        return
    if (forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0):
        self.req_to_token_pool.mamba_pool.clear_slots(
            forward_batch.mamba_clear_indices)
    ...
    forward_batch.mamba_clear_indices = None
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:影响所有启用 mamba extra buffer / hybrid linear attention 的路径(hybrid_linear_attn_backend、Mamba radix cache 两个变体、unified cache mamba component、EAGLE verify 的 track indices 构建)。行为变化点:mamba 槽位分配不再即时清零,清零与 COW 都延迟到 forward 流的 extend 批次;若某路径分配槽位后未进入 extend 前向,槽位将保持未清零状态(由 diff 推断)。draft worker 明确跳过延迟操作。decode 路径的 track indices 构建逻辑等价重构。

## 关联
- [issue #24221 评论](https://github.com/sgl-project/sglang/issues/24221#issuecomment-4424945325)(0kl RL 竞态验证上下文)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
