# PR #29792: Fix Mamba track-boundary bookkeeping under overlap scheduling

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @huangtingwei9988 |
| 合并 | 2026-08-11 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | high priority, hicache, run-ci, apple-silicon, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/29792 |

## 一句话结论

修正 overlap 调度下 Mamba track 边界与 ping-pong 槽的归属错误,消除缓存重放后 KL 发散。

## 动机

Mamba 每次 decode forward 用 batch 内 `seq_lens_cpu` 快照构建 `mamba_track_mask` 决定是否记录 recurrent state,但 result processing 侧却用 `Req.kv_committed_len`(共享可变请求状态)判断 track 边界。overlap 调度会先 launch 下一批再处理上一批结果:处理 batch A 结果时,同一 Req 对象可能已被 batch B 的 `prepare_for_decode()` 推进(如 kv_committed_len 从 127 到 128),于是 A 被误判为"越过了 track 边界",对错误的 batch 做了 bookkeeping。在 lazy extra-buffer 模式下,这可能推进/释放/更新错误的 ping-pong track 槽、把不对应 A forward 的状态记进 `mamba_last_track_seqlen`,后续 cache-hit 重放会加载与缓存前缀不一致的 Mamba state,导致 KL 发散回归。PR 正文还专门论证了"在 forward stream 上加 wait 能让测试通过"只是同步掩蔽(wait 后 B 的 forward 通常已写完,错误路径恰好读到完成态),并不修复语义,且会拖慢 result-processing 路径。

## 优化点

两处归属修正。其一,boundary 判定改用与 forward 一致的 per-batch 快照:非 spec 路径用 `committed_len = req.kv_committed_len - lookahead`(`lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]`,限 0/1)复现 forward 侧的 `seq_lens_cpu % interval == 0` 检查,消除共享可变字段被下一批提前推进的影响。其二,拆分 `mamba_next_track_idx` 的双重语义:新增 producer 侧 `req.mamba_next_track_idx`(下个写入槽)与 consumer 侧 `req.mamba_last_track_idx`(最近有效状态槽);batch 级新增 `mamba_track_buffer_indices` 快照记录本次 forward 实际选中的 ping-pong 位置,result processing 据此更新 `mamba_last_track_idx`;`get_mamba_ping_pong_keep_idx`(缓存插入/donate 用)统一改读 `mamba_last_track_idx`,lazy 模式不再把"下一个写入目标"当作"最近有效状态"捐给前缀缓存。free/retract/merge/copy/session 恢复等所有生命周期路径同步维护新字段。

## 关键改动

```text
python/sglang/srt/managers/schedule_batch.py: Req 新增 mamba_last_track_idx;
  ScheduleBatch 新增 mamba_track_buffer_indices 快照;track 时同步记录
python/sglang/srt/managers/scheduler_components/batch_result_processor.py:
  _mamba_check_track_boundary 减去 overlap lookahead;
  _mamba_prefix_cache_update/_mamba_lazy_spec_update 用快照更新 last_track_idx
python/sglang/srt/mem_cache/memory_pool.py:
  get_mamba_ping_pong_keep_idx 改为返回 req.mamba_last_track_idx
python/sglang/srt/session/streaming_session.py, hardware_backend/mlx/...:
  SessionSlot 与 MLX 路径同步保存/恢复/清理新字段
```

```python
# result processing:用 per-batch 快照复现 forward 的边界检查
lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
committed_len = req.kv_committed_len - lookahead
if committed_len % interval == 0:
    return True, committed_len
```

## 性能声明

PR 正文未提供性能数字(Accuracy/Speed 小节为空,checklist 未勾选)。本 PR 是正确性修复;正文指出的性能意义在于:替代方案"result-processing 前对 forward stream 加 wait"是全局同步、会损害性能,本修复去掉了对该 workaround 的需求。

## 验证与回归风险

测试改动:`test_unified_radix_cache_kl_mamba.py` 的 hicache-io-backend 由 direct 改 kernel、mem-layout 由 page_first_direct 改 page_first(即 KL 测试运行在 kernel IO 布局下)。回归风险:新字段 `mamba_last_track_idx` 在 free/retract/merge/copy/session save/restore/MLX 后端共 10+ 处生命周期点必须一致维护,遗漏任一处会重现错误槽 keep;lookahead 限定 0/1,更深 overlap 流水会破坏减法修正;lazy 模式下 finished req 的 `mamba_lazy_is_insert` 分支语义变化需与 prealloc 失败场景兼容。

## 关联

同区域(Mamba + overlap 调度):本批次索引中 #14792([Qwen3-next] support mamba radix cache for overlap scheduler)为同一交叉区域的前序工作;pack 未列出其他直接关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
