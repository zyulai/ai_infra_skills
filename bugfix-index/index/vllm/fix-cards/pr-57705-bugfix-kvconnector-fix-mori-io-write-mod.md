# PR #57705: [Bugfix][KVConnector] Fix MoRI-IO WRITE-mode requests stranded in WAITING_FOR_REMOTE_KVS

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57705](https://github.com/vllm-project/vllm/pull/57705) |
| 作者 | akshayv |
| 合并日期 | 2026-10-12 |
| labels | bug, documentation, kv-connector, ready, rocm |
| 热度 | 👍 0 · 💬 29 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
MoRI-IO WRITE(push)模式下 decode 过载时请求永久滞留 WAITING_FOR_REMOTE_KVS;修复 prefill 超时静默弃写、全前缀命中发空块表两个根因。

## 问题现象
MoRI-IO WRITE(push)模式下,decode 过载时 decode 请求会永远卡在 `WAITING_FOR_REMOTE_KVS`。引擎本身保持健康、继续服务新请求,因此表现为 benchmark 永远跑不完而非报错。复现:1024 输入/1024 输出、128 个请求、并发 64,decode 限 20 个运行请求(`--max-num-seqs 20`)时,`main` 分支在全部请求完成前 stall。

## 根因
正文披露两个根因(第三个根因 decode preemption 已由 #53250 修复,本 PR 移除了与之重复的改动):
1. **Prefill 放弃延迟写但不通知 decode**:push 模式下代理把每个请求同时发给 prefill 和 decode;prefill 算完 KV 后等 decode 分配块并回传 ID。decode 忙时,请求在 decode 队列等待可超过 `defer_timeout`(60 s)。`main` 上 prefill 超时后释放自己的块并丢弃写入,但从不告知 decode;decode 之后放行该请求时,等待永远不会到来的 KV。
2. **全量本地前缀缓存命中发送空块表**:decode 已有整个 prompt 时给 prefill 发了空块列表,prefill 以 `MoRIIOError("block_notify_list cannot be empty")` 拒绝并保留自己的块。

## 修复方案
- 超时处理重做(`moriio_engine.py`/`moriio_connector.py`/`moriio_common.py`):延迟写超时先在锁内调用 `fail_unallocated_write` 把 transfer 标记为终态(decode 迟到的块分配会被忽略、写入无法启动;若分配恰好在超时前到达则照常执行),再向每个 decode DP rank 广播 `write_failed`(prefill 不知道请求归属哪个 rank,非属主 rank 忽略;未匹配失败以 `_MAX_UNMATCHED_WRITE_FAILURES=4096` 封顶),然后释放 prefill 块。decode 侧通过新增的 `get_transfer_results` 把 `write_failed` 报告为 KV load 失败,按 `kv_load_failure_policy` 失败或重算;因从未发生写入,decode 可立即释放块。prefill scheduler 不再自行用 WRITE 模式超时释放块(`update_connector_output` 的过期回收仅限 READ 模式),只有 worker 在 transfer 终态后释放。
- 全前缀命中修复:`update_state_after_alloc` 在 `num_external_tokens == 0` 时改发 release 消息(`_release_write_prefill_blocks`),不再发被拒绝的空分配通知。
- 辅助重构:`WriteTask` 增加 `remote_dp_size`,抽出 `_resolve_notify_endpoint` 统一 multi-pod host/port 解析。
- 新增 3 个 CPU 单测覆盖超时广播、scheduler 等待 worker ACK、失败上报。

```python
+            elif (
+                now - task.enqueue_time > defer_timeout
+                and self.worker.moriio_wrapper.fail_unallocated_write(task.transfer_id)
+            ):
+                self._notify_write_failed(task, now - task.enqueue_time)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:MoRI-IO connector 的 WRITE 模式全链路(prefill scheduler/worker/engine wrapper、decode 消费端)。READ 模式的 deferred-send 回收逻辑保留不变;非 WRITE 失败路径(如 #58962 覆盖的 prefill 中途死亡)不受影响。验证:207 个 MoRI-IO 单测通过(main 上 204 + 新增 3);端到端(2×Radeon AI Pro R9700S、Llama-3.1-8B-Instruct、配合 #53250)benchmark 完成、双端 KV cache 归零——`fail` 策略 109 成功/19 失败,`recompute` 策略 128 全成功/34 次超时重算。

## 关联
- #53250(decode preemption 根因的修复,本 PR 依赖)
- #58962(prefill 中途失联时 decode 侧接收超时,与本 PR 互补)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
