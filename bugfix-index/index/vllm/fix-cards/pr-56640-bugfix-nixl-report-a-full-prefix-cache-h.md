# PR #56640: [Bugfix][NIXL] Report a full prefix cache hit as finished

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56640](https://github.com/vllm-project/vllm/pull/56640) |
| 作者 | jyizheng |
| 合并日期 | 2026-09-15 |
| labels | bug, ready, ci/build, kv-connector |
| 热度 | 👍 0 · 💬 28 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
NIXL 全前缀缓存命中时 `_read_blocks` 提前返回且不记录,请求卡死在 skipped_waiting 永久占用 KV 块;以 awaiting_kvs 门控上报空 transfer 列表释放请求。

## 问题现象
10-prefill/3-decode 饱和部署中:对端重启导致 `send_notif` 半小时内失败 122 次,这些请求全部被搁置;3 个 decode worker 在零 running、零 waiting、无任何流量的情况下仍占用 73–85% 的 KV cache,直到 liveness probe 杀掉进程,只有重启能清除。该状态从外部难以观察:请求停在 `skipped_waiting`,两个队列深度指标都读零,但大部分缓存已被占用。

## 根因
正文披露:本地 block 列表为空时(即 D 侧已持有全部 KV 的全前缀缓存命中),`_read_blocks` 提前返回——只需通知生产者——但返回时不记录任何东西;而 `get_finished` 仅从 `_recving_transfers` 与 `_failed_recv_reqs` 构建 `done_recving`,该请求两边都不在。调度器对 `WAITING_FOR_REMOTE_KVS` 请求的唯一释放途径是 `finished_recving_kv_req_ids`,且没有超时机制,于是请求永久滞留 `skipped_waiting` 并持有已分配的 block,直至进程结束。

## 修复方案
为 recv 增加 `awaiting_kvs` 门控,仅"被调度器挂起等待"的请求上报完成:
- `metadata.py`:`ReqMeta`/`add_new_req_to_recv` 新增 `awaiting_kvs: bool = False`。
- `base_scheduler.py`/`pull_scheduler.py`/`push_scheduler.py`:`_reqs_need_recv` 元组扩为 4 元素;pull 侧 `update_state_after_alloc` 以 `num_external_tokens > 0` 置位(有外部 token 才 parked,全本地命中保持 RUNNING);notify-only 路径一律 False。
- `pull_worker.py::_read_blocks`:通知失败路径上,若 `awaiting_kvs` 则记录空 transfer:

```python
if awaiting_kvs:
    self._recving_transfers.setdefault(request_id, [])
```

`_pop_done_transfers` 本就把无未完成句柄的请求视为完成。新增测试覆盖通知成功/失败 × awaiting_kvs True/False,断言仅 awaited 请求进入 `done_recving`(notify-only 上报会触发调度器断言)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:NIXL connector 的 pull/push scheduler 与 pull worker 元数据契约(元组结构、ReqMeta 字段);行为变化集中在"parked 的全前缀命中请求现在会出现在 done_recving",notify-only recv 保持不上报,否则引擎核心断言失败(由 diff 推断)。

## 关联
- #54714(被错误的 force-push 意外关闭,本 PR 重新打开)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
