# PR #29792: Fix Mamba track-boundary bookkeeping under overlap scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29792](https://github.com/sgl-project/sglang/pull/29792) |
| 作者 | huangtingwei9988 |
| 合并日期 | 2026-08-11 |
| labels | high priority, hicache, run-ci, apple-silicon, bypass-fastfail, run-ci-extra |
| 热度 | 👍 3 · 💬 70 |
| 分类 | Overlap / Scheduler |
| bug 类型 | general |

## 一句话结论

overlap 调度下结果处理误用共享的 `kv_committed_len` 判断 Mamba track 边界并混用 `mamba_next_track_idx` 双重语义,导致缓存插入保留错误 slot、KL 回归;改为用 batch 快照并新增 `mamba_last_track_idx`。

## 问题现象

Mamba 的每次 decode forward 通过 batch 内 `seq_lens_cpu` 快照构建 `mamba_track_mask`,决定是否记录 recurrent state;但结果处理侧此前用 `Req.kv_committed_len`(共享可变请求状态)检查 track 边界。overlap 调度下 scheduler 在处理上一批结果之前就会 prepare 并启动下一批,因此处理 batch A 的结果时,同一 Req 对象可能已被 batch B 的 `prepare_for_decode()` 推进。正文给出的例子(`mamba_track_interval`=128):batch A forward 时 `seq_len`=127,mask 为 false,不写 tracked state;A 结果尚未处理时 batch B 已被 prepare,`kv_committed_len` 从 127 推到 128,B 的 mask 为 true、应由 B 写 tracked state;随后 A 的结果处理读到 128,误判 A 越过了 track 边界。于是 Mamba post-decode bookkeeping 跑在错误的 batch 上。lazy extra-buffer 模式下,可能对 ping-pong track slot 做 advance/free/update,并把与 batch A forward 不对应的状态记入 `mamba_last_track_seqlen`;后续 cache-hit replay 加载到与缓存前缀不一致的 Mamba state,造成 KL divergence 回归。正文还解释了为什么加 forward-stream wait 能让测试通过:wait 只是同步 workaround,错误的批次仍被处理,只是观察到已完成状态的概率更高;wait 还会损害性能。

## 根因

正文披露两点。其一,结果处理侧用共享可变的 `Req.kv_committed_len` 判断边界,而 forward 用的是 batch-local 快照;overlap 下该字段在结果处理前已被下一批 `prepare_for_decode()` 推进,两个视角错位。其二,`req.mamba_next_track_idx` 被用于两种不同含义:未来 forward 应写入的下一个 ping-pong slot,以及当前保存最后有效 Mamba state、应保留/捐赠给 prefix cache 的 slot。lazy extra buffer 下,后一个 batch 会在上一批结果处理并插入缓存之前,于 prepare/prealloc 阶段更新 `mamba_next_track_idx`,导致缓存插入调用 `get_mamba_ping_pong_keep_idx(req)` 时保留的是未来的 prealloc slot,而不是真正持有最后写入 Mamba state 的 slot。cache-hit loadback 恢复出数值错误的 state,模型能跑但 KL 不稳定或超阈值。

## 修复方案

让结果处理与 forward 使用同一 per-batch 快照,并显式区分两个指针:

- `_mamba_check_track_boundary` 改为 `committed_len = req.kv_committed_len - lookahead`(lookahead 来自 `batch.mamba_decode_batch_idx_cpu`),复现 forward 的 `% interval == 0` 检查;边界判定改读 `batch.mamba_track_mask_cpu[i]`(completed boundary)。
- 新增 `req.mamba_last_track_idx`(consumer 侧,最后有效 slot)与 `mamba_next_track_idx`(producer 侧)分离;`get_mamba_ping_pong_keep_idx` 直接返回 `mamba_last_track_idx`。
- 新增 per-batch 快照 `batch.mamba_track_buffer_indices`,在 filter/merge/copy 中同步维护;结果处理用它更新 `req.mamba_last_track_idx`,避免两个 req 级索引在 overlap 下同时推进造成错位。
- 重写 `mamba_lazy_post_decode_at_boundary(req, batch, track_idx)` 为「提交完成的 lazy 边界并释放旧 slot」;`_mamba_prefix_cache_update`、`_mamba_lazy_spec_update` 相应改为写 `mamba_last_track_idx`。

关键摘录(memory_pool.py):

```diff
-        if self.enable_mamba_extra_buffer_lazy:
-            return req.mamba_next_track_idx
-        return self.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)
+        return req.mamba_last_track_idx
```

改动文件:`schedule_batch.py`、`scheduler_components/batch_result_processor.py`、`mem_cache/memory_pool.py`、`hardware_backend/mlx/kv_cache/auxiliary_state.py`、`hardware_backend/mlx/model_runner.py`、`session/streaming_session.py`,并将 KL 测试的 hicache io backend 从 direct/page_first_direct 改为 kernel/page_first。

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围,不写。
- 回归面:按 diff,Mamba / hybrid GDN 模型在 overlap 调度下的 track 边界判定、lazy extra buffer 的 ping-pong slot 分配与释放、prefix cache 插入与捐赠路径均被重写;`mamba_last_track_idx` 作为新状态字段贯穿 Req 初始化、retract、free/cleanup、session 保存恢复与 MLX backend;KL 测试参数(io backend / mem layout)也改变。普通非 Mamba 模型路径不受影响。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
