# PR #12368: [Bug fix][PD Dissaggregation] fix prefill hanging issue with PP and DP Attention,

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12368](https://github.com/sgl-project/sglang/pull/12368) |
| 作者 | popsiclexu |
| 合并日期 | 2025-10-30 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
PD 分离 + PP + DP Attention 下 prefill 挂起:disagg prefill 事件循环用了完整 TP 组做同步且缺 mlp-sync 批准备;改为 attention TP 子组并补上 `prepare_mlp_sync_batch`。

## 问题现象
标题描述的现象:PD Disaggregation 场景下,同时启用 PP 与 DP Attention 时 prefill 阶段挂起(hang)。正文 Motivation 仅写 "fixed #12121",未描述具体复现步骤、日志或影响面。

## 根因
正文未披露根因(Modifications 一节为空,仅说 fixed #12121)。(由 diff 推断) 挂起与两组同步点有关:
1. `disaggregation/prefill.py` 中,`get_transferred_rids` 的 `poll_and_all_reduce` 用的是完整 `tp_group.cpu_group`,`recv_pyobj_from_prev_stage` 的 `broadcast_pyobj` 也以 `tp_size != 1` 为条件、在完整 TP 组上广播——在 DP Attention 下该集合通信应发生在 attention TP 子组(attn_tp_group),用错误规模的组做 all-reduce/broadcast 会使不参与该通信的 rank 等不到对端而挂起;
2. `scheduler_pp_mixin.py` 的 `event_loop_pp_disagg_prefill` 中,`get_new_batch_prefill()` 之后没有做 `prepare_mlp_sync_batch`,而在 `require_mlp_sync(server_args)` 为真的配置下各 PP stage / DP rank 的批次未经同步,后续推进节奏不一致导致卡住。

## 修复方案
- `python/sglang/srt/disaggregation/prefill.py`:
  - `get_transferred_rids`:`poll_and_all_reduce` 的通信组从 `self.tp_worker.get_tp_group().cpu_group` 改为 `self.tp_worker.get_attention_tp_cpu_group()`;
  - `recv_pyobj_from_prev_stage`:广播条件从 `self.tp_size != 1` 改为 `self.attn_tp_size != 1`,改用 `self.attn_tp_group.rank`、`self.attn_tp_cpu_group`、`src=self.attn_tp_group.ranks[0]`。
- `python/sglang/srt/managers/scheduler_pp_mixin.py`:`event_loop_pp_disagg_prefill` 中在 `get_new_batch_prefill()` 后补上条件同步:

```diff
-                mbs[mb_id] = self.get_new_batch_prefill()
+                batch = self.get_new_batch_prefill()
+                if require_mlp_sync(self.server_args):
+                    batch = self.prepare_mlp_sync_batch(batch)
+                mbs[mb_id] = batch
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `disaggregation/prefill.py` 与 `scheduler_pp_mixin.py` 两个文件。影响 PD 分离 prefill 路径:在 attn_tp_size < tp_size 的配置(DP Attention)下,传输确认与跨 stage 广播改在更小的 attention TP 子组上进行;`tp_size == attn_tp_size` 的普通配置下两组一致,行为不变(由 diff 推断)。PP disagg prefill 事件循环中,`require_mlp_sync` 为真的配置会新增一次 `prepare_mlp_sync_batch` 调用,其他配置无额外开销。非 PD 分离路径不受影响。

## 关联
- Fixes sgl-project/sglang#12121(正文提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
