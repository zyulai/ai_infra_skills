# PR #24461: Revert "[fix] /pause_generation and /continue_generation wrong for --tokenizer-worker-num > 1"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24461](https://github.com/sgl-project/sglang/pull/24461) |
| 作者 | merrymercy |
| 合并日期 | 2026-05-05 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论
#24445 的 pause/continue 广播修复首次 CI 失败,整体回退该 PR,恢复 `--tokenizer-worker-num > 1` 时多 tokenizer 路由的原始行为。

## 问题现象
本 PR 为回退 PR,针对 #24445「[fix] /pause_generation and /continue_generation wrong for --tokenizer-worker-num > 1」。正文说明回退原因:被回退的修复"failed the first CI",并附失败 run 链接 https://github.com/sgl-project/sglang/actions/runs/25408313988/job/74524211822?pr=24458#step:7:70 。被回退修复原本要解决的问题是 `--tokenizer-worker-num > 1` 时 `/pause_generation` 与 `/continue_generation` 行为错误;除此之外本 PR 正文未描述更多具体现象。

## 根因
正文未披露被回退代码的技术根因,仅说明 #24445 首次 CI 失败因此回退。(由 diff 推断) 被回退的机制是:为支持多 tokenizer worker 下的 pause/continue,`io_struct.py` 新增 `TokenizerWorkerRegisterReq` 与 `PauseContinueBroadcast` 两个 dataclass;每个 TokenizerWorker 启动时向 router 注册自己的 IPC 名,router 在 `router_worker_obj` 转发路径上拦截 pause/continue 请求、经 `SocketMapping` 向所有已注册 worker 广播 `PauseContinueBroadcast`(abort 模式不转发给 scheduler),worker 侧在 result dispatcher 中挂入 `_handle_pause_continue_broadcast`/`_apply_pause_continue_broadcast`,用 future + `is_pause_cond` 同步各 worker 的 is_pause 状态。该链路在 CI 中失败后,本 PR 将上述注册、广播与状态同步代码全部删除。

## 修复方案
整体 revert #24445,涉及两个文件:
- `python/sglang/srt/managers/io_struct.py`:删除 `TokenizerWorkerRegisterReq`、`PauseContinueBroadcast` 两个 dataclass;
- `python/sglang/srt/managers/multi_tokenizer_mixin.py`:
  - `MultiTokenizerRouter` 恢复原 docstring,删除 worker IPC 注册集合 `all_worker_ipcs`,把 `socket_mapping` 的创建移回 `handle_loop`;
  - `router_worker_obj` 删除 pause/continue 拦截与广播逻辑,恢复为纯转发(`await self.send_to_scheduler.send_pyobj(recv_obj)`);
  - worker 侧删除注册请求、`_pause_continue_future`、`pause_generation`/`continue_generation` 覆写及 `_handle_pause_continue_broadcast`/`_apply_pause_continue_broadcast`。

diff 摘录(router 转发路径恢复):
```diff
         recv_obj = await self.receive_from_worker.recv_pyobj()
-
-            if isinstance(recv_obj, TokenizerWorkerRegisterReq):
-                ...
-                continue
-
-            if isinstance(
-                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
-            ):
-                ...broadcast...
-                continue
-
             await self.send_to_scheduler.send_pyobj(recv_obj)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `python/sglang/srt/managers/io_struct.py` 与 `multi_tokenizer_mixin.py`。影响多 tokenizer worker(--tokenizer-worker-num > 1)部署下 `/pause_generation`、`/continue_generation` 的处理路径:回退后这些请求不再经 router 广播同步各 worker 的 is_pause 状态,即原问题「pause/continue 在多 worker 下行为不对」会重新存在;单 worker 与其他请求的转发、结果分发路径恢复原实现。

## 关联
- Reverts sgl-project/sglang#24445
- https://github.com/sgl-project/sglang/actions/runs/25408313988/job/74524211822?pr=24458#step:7:70 (失败的 CI run)
- https://github.com/sgl-project/sglang/blob/431ca543345ec45476aaea79c99dd6c18a604cdc/.github/MAINTAINER.md (正文引用的维护者规则)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
