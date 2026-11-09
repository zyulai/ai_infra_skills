# PR #56841: [Bugfix][KVConnector] Make ExampleHiddenStatesConnector abort-safe

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56841](https://github.com/vllm-project/vllm/pull/56841) |
| 作者 | ys2025-AI |
| 合并日期 | 2026-09-18 |
| labels | bug, ready, kv-connector |
| 热度 | 👍 0 · 💬 14 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
排队中请求被 abort 时 ExampleHiddenStatesConnector 无条件 pop 文件名抛 KeyError 打崩 EngineCore;pop 加默认值等四处防御修复。

## 问题现象
请求在 waiting(从未被调度)时被 abort——API 客户端超时后取消排队请求的常见情形——`request_finished()` 抛出 `KeyError: 'cmpl-b98ee80d8ff99a38-0-8e7006c7'`,变成「EngineCore encountered a fatal error」,所有 API server 进程死亡,客户端看到 `APIConnectionError` 风暴。E2E:vLLM 0.26.0 + vllm-ascend(Ascend NPU)在线 DSpark hidden-state training,abort 风暴下此前 EngineCore 必崩,修复后训练跑完。复现方式:发一批长 prompt 请求,让部分超时并在排队中被 abort。

## 根因
正文披露:`_request_filenames` 只在 `build_connector_meta()` 处理 `SchedulerOutput.scheduled_new_reqs` 时填充;从未调度的请求不会到达该代码,但 `Scheduler.finish_requests()` 仍会对 waiting 请求调用 `request_finished()`,其中无条件的 `self._request_filenames.pop(req_id)` 抛 KeyError。

## 修复方案
`example_hidden_states_connector.py` 四处:① `request_finished` 改 `pop(req_id, None)`,无 filename 时清理 `_pending_saves` 残留并返回 (False, None);② `request_finished_all_groups` 在 per-group block table 缺失时 no-op 而非越界索引;③ `get_finished` 只为实际跟踪过(已提交 DtoH copy)的请求上报 finished_sending,避免 abort/未调度请求触发 `_update_from_kv_xfer_finished()` 的 assert;④ `get_finished_count()` 返回 1——只有 TP rank 0 写 hidden states,默认 None 会让 `KVOutputAggregator` 按 world_size 等待、请求永不完成并泄漏 KV blocks。③④ 耦合:

```diff
-        filename = self._request_filenames.pop(req_id)
+        filename = self._request_filenames.pop(req_id, None)
+        if filename is None:
+            self._pending_saves.pop(req_id, None)
+            return False, None
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本(E2E 验证于 vLLM 0.26.0)。
- 回归面:该示例 connector 的 abort/收尾路径(request_finished、request_finished_all_groups、get_finished)与完成通知计数;正常完成路径的 finished_sending 语义不变,未跟踪请求不再被上报;新增 CPU-only 单测覆盖 5 个场景。

## 关联
- https://github.com/vllm-project/vllm/issues/46240
- https://github.com/vllm-project/vllm/pull/46304
- https://github.com/vllm-project/vllm/pull/53079

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
