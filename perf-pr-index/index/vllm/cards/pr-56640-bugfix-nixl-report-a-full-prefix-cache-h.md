# PR #56640: [Bugfix][NIXL] Report a full prefix cache hit as finished

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jyizheng |
| 合并 | 2026-09-15 |
| 分类 | Overlap / Scheduler(tier: major) |
| Label | bug, ready, ci/build, kv-connector |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(7 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/56640 |

## 一句话结论

修复 NIXL KV connector 的资源泄漏:全前缀缓存命中不被上报为 finished,请求永久滞留 WAITING_FOR_REMOTE_KVS 并占住 KV cache 块——真实事故中 3 个 decode worker 在零流量下持 73-85% KV cache 直至被 liveness probe 杀掉。

## 动机

`_read_blocks` 在本地 block 列表为空时提前返回(decode 侧 D 已持有 KV,只需通知 producer)且不记录任何 transfer;`get_finished` 只从 `_recving_transfers` 与 `_failed_recv_reqs` 构建 `done_recving`,该请求两边都不在。Scheduler 对 `WAITING_FOR_REMOTE_KVS` 请求只认 `finished_recving_kv_req_ids` 且无超时,请求停在 `skipped_waiting`(两个队列深度 gauge 都不计入)占着已分配块直到进程结束。实际故障:10-prefill/3-decode 部署在饱和负载下,peer 重启使 `send_notif` 半小时失败 122 次,每个此类请求被弃置;3 个 decode worker 在 zero running / zero waiting、部署无任何流量的状态下持有 73-85% KV cache,只有重启能清除。

## 优化点

引入 `awaiting_kvs` 标志贯穿 scheduler → metadata → worker:`update_state_after_alloc` 仅在 `num_external_tokens > 0`(scheduler 真正把请求 park 进 WAITING_FOR_REMOTE_KVS)时置 True;`request_finished` 为释放 P 的块而 seed 的 notify-only 空 recv 置 False;push scheduler 一律 False。`_read_blocks` 的提前返回路径上,即使通知发送失败,只要 `awaiting_kvs` 为 True 就执行 `self._recving_transfers.setdefault(request_id, [])` 上报——空 transfer 列表已足以让 `_pop_done_transfers` 把该请求视为完成(通知失败只让 producer 超时后自行释放块)。notify-only recv 必须保持不上报,否则 scheduler `_update_from_kv_xfer_finished` 中 `assert req_id in self.requests` 会崩引擎。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py: _read_blocks 提前返回路径按 awaiting_kvs 记录空 transfer(通知失败也上报)
vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py: update_state_after_alloc 按 num_external_tokens>0 置 awaiting_kvs
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py: _reqs_need_recv 元组扩展为含 awaiting_kvs 并写入 metadata
vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py: ReqMeta 新增 awaiting_kvs 字段
vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py: 两处空 recv 显式置 False
```

## 性能声明

无性能数字;为可用性/资源泄漏修复。事故量化数据(30 分钟内 send_notif 失败 122 次;decode worker 持有 73-85% KV cache、zero running/zero waiting)来自 PR 正文描述的生产部署,修复后该类请求能正常完成并释放块。

## 验证与回归风险

`tests/v1/kv_connector/unit/test_nixl_connector.py` 新增 `test_empty_recv_is_reported_only_when_awaited`:按 awaiting_kvs(True/False)× notification_fails(True/False)四组合参数化,断言仅 parked 请求进入 `done_recving`;`test_full_local_hit_is_not_awaited_by_the_scheduler` 固化 num_external_tokens==0 时不置 awaiting;abort-seed 测试断言 `awaiting_kvs is False`。回归风险:上报行为被精确限定于 scheduler park 的请求,notify-only 上报会触发 scheduler assert(测试固化该边界);HMA 测试同步更新元组解包。

## 关联

#54714(同一 commit 的原 PR,因 bad force-push 孤儿分支被意外关闭后重开;含 @njhill 的 review 意见)。同分类 Overlap / Scheduler 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
