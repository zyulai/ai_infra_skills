# PR #50147: [Fix] dp-attention: Deliver an abort behind its request on every DP rank

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#50147](https://github.com/sgl-project/sglang/pull/50147) |
| 作者 | yuxuandexter |
| 合并日期 | 2026-11-15 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论

开启 DP attention 且使用默认控制路由时,`AbortReq` 不在 `DataParallelController` 的分发表里、只能走控制消息通道被发到 DP rank 0,再经 rank 0 的广播覆盖其他 rank 的控制列表,abort 与请求本体之间没有顺序保证,请求可能先被调度执行导致 abort 丢失;修复是把 `AbortReq` 路由为 `send_to_all_workers` 并让它随 work 请求通道按 socket 顺序在 DP 组内广播。

## 问题现象

Fixes #50137。在数据并行 attention + 默认控制路由(`--enable-dp-attention-local-control-broadcast` 关闭)下,正在运行的请求的 abort 有时被丢失:`POST /abort_request` 返回 200,tokenizer 也转发了 abort,但请求一直跑到 `max_new_tokens` 并以 `length` 结束。

复现测量(commit `d800cb859e85`,两块 GB200,Qwen/Qwen3-0.6B,`--tp-size 2 --attn-dp-size 2 --enable-metrics --mem-fraction-static 0.5 --context-length 4096`,受害请求固定到 DP rank 1、rank 0 上保持 128 条积压、提交后 0–20ms 内 abort):

- 200 次 abort 丢失 19 次(Wilson 95%:6.2%–14.4%);按延迟分组 0ms 10/34、1ms 5/34、2ms 2/33、5ms 0/33、10ms 0/33、20ms 2/33。
- `SGLANG_SCHEDULER_MAX_RECV_PER_POLL=1` 且受害 rank 上积压 256 条时:16/16 全部丢失。
- 对照组:`--enable-dp-attention-local-control-broadcast` 开启时 0/100 与 0/16 丢失;受害请求在 DP rank 0 时 0/12 丢失。

## 根因

正文披露:`DataParallelController` 把 `TokenizedGenerateReqInput` 发到所选 DP rank 自己的 socket,但 `AbortReq` 在分发表中没有条目,回退到 `send_control_message`,该路径在 DP attention + 默认控制路由下发到 `self._active_workers[::tp_size]`,即只有 DP rank 0。rank 0 的 leader 又把控制列表经整个 `tp_group` 广播,`_broadcast_reqs_across_ranks` 会用该广播覆盖其他 leader 的控制列表。于是请求的 owner 一边在自己的 socket 上收到请求、一边通过 rank 0 的广播收到 abort,两者没有先后顺序;当 owner 恰在请求到达前排空 socket 时会先应用 abort,`Scheduler.abort_request` 找不到该 rid 也不留记录,请求随后入队并跑完。tokenizer 不会重发 abort(见 `_skip_dispatch_if_aborted` 的 docstring)。该交错由双 socket 控制路由的模型推导得出,并在引擎上复现。

## 修复方案

1. `DataParallelController.init_dispatcher`:把 `AbortReq` 路由到 `send_to_all_workers`(与 `BlockReqInput`、`ProfileReq` 一致)。每个 DP rank 都在自己接收请求的 socket 上收到 abort,ZeroMQ 保证 abort 排在 tokenizer 先发的请求之后。DP attention 关闭或 local control broadcast 开启时,`control_message_step` 已是 1,abort 路由与之前相同。
2. `SchedulerRequestReceiver._split_work_and_control_reqs`(仅 DP attention 使用):把 `AbortReq` 与 work 请求放在一起、按 socket 顺序保留,并在其 DP 组内广播。否则 owner socket 上排在请求后的那份副本仍会被 rank 0 的控制广播覆盖。

正文强调两个改动缺一不可:只改分发表,owner 自己的副本仍被 rank 0 广播覆盖;只改 receiver,abort 出不了 DP 组 0。

关键 diff 摘录:

```diff
                 (BlockReqInput, self.send_to_all_workers),
                 (ProfileReq, self.send_to_all_workers),
+                # A scheduler ignores an abort for a rid it has not received yet,
+                # so every DP rank gets the abort on the socket its requests use.
+                (AbortReq, self.send_to_all_workers),
```

```diff
                     TokenizedEmbeddingReqInput,
                     BatchTokenizedGenerateReqInput,
                     BatchTokenizedEmbeddingReqInput,
+                    AbortReq,
                 ),
             )
         ]
```

新增 CPU 单测 `TestDPAttentionAbortRoute.test_abort_is_applied_after_its_request_on_the_owner_rank`,驱动真实分发表与 `_broadcast_reqs_across_ranks` 走丢失交错,验证 owner 先应用请求再应用 abort、每个 DP 组恰好应用一次 abort。

## 影响与回归风险

- GPU 复现验证:默认路由 200 次全送达(修复前丢 19),全部以 `abort` 结束、`/abort_request` 返回后最多再出 2 个 token;`SGLANG_SCHEDULER_MAX_RECV_PER_POLL=1` 场景 16/16 送达;单元测试 31 个全过(修复前 1 个失败)。
- 可见顺序变化(仅 DP attention):同一次接收内,`AbortReq` 现在先于该次接收的控制消息应用(work 请求本就如此);与 `PauseGenerationReqInput` 同批拉到的 abort 会立即生效而非被暂停挡住。
- 不变部分:每个 DP 组此前已通过 rank 0 广播应用过每个 abort,现在改为从自己的 socket 应用,`abort_all` 与不持有 rid 的 rank 行为不变;调度器对未收到的 rid 仍保持无状态(#35255、#47546 的原则)。性能上仅在发出 abort 时多发几个小 `AbortReq`,无新增 collective。
- 正文用三棵树对照(未打补丁/已打补丁/回滚两文件)排除了环境干扰;Extra 与 AMD CI 在该 PR 上为失败状态,Base CI 通过。

## 关联

- Fixes #50137。
- 相关原则引用:#47532(Rust server 对 abort 的处理)、#47546、#35252 相关的调度器无状态假设、#47693(“FIFO ordering” 假设在此路由上对 abort 成立)。
- 新增测试:`test/registered/unit/managers/test_data_parallel_controller.py`;相关 receiver 路由测试 `test_pp_cp_rank_offsets.py` 5 个通过。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
