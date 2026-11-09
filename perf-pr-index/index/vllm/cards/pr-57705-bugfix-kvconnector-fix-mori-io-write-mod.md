# PR #57705: [Bugfix][KVConnector] Fix MoRI-IO WRITE-mode requests stranded in WAITING_FOR_REMOTE_KVS

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @akshayv |
| 合并 | 2026-10-12 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | bug, documentation, rocm, ready, kv-connector |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/57705 |

## 一句话结论

修复 MoRI-IO WRITE(push)P/D 分离模式下,decode 过载时请求永久卡在 WAITING_FOR_REMOTE_KVS 的挂死问题。

## 动机

MoRI-IO WRITE 模式下,proxy 同时把请求发给 prefill 与 decode:prefill 算完 KV 后等 decode 分配块并回传块 ID。当 decode 过载(如 1024 in/1024 out、128 请求、并发 64、decode 限 20 running),请求在 decode 队列里等待可能超过 `defer_timeout`(60 s)。此时 prefill 在 main 上静默释放块并放弃写入但不通知 decode;decode 之后放行该请求时,等待永远不会到来的 KV,请求卡死。引擎本身仍健康并继续接新请求,表现为 benchmark 永不结束而非报错。

## 优化点

修两个根因。其一,deferred write 超时走显式失败协议:prefill 侧 `fail_unallocated_write()` 在锁内把 transfer 标记为 terminal(若 decode 的块分配恰好已到则正常执行写入),随后向所有 decode DP rank 广播 `write_failed` 消息(prefill 不知道请求归属哪个 rank,非归属 rank 直接忽略,未匹配的失败记录按 `_MAX_UNMATCHED_WRITE_FAILURES=4096` 有界保留),然后才释放块;decode 侧把该 transfer 映射回请求,经 `get_transfer_results()` 报为 failed KV load,按 `kv_load_failure_policy` 走失败或本地重算,因写入从未发生可立即释放块。同时 prefill scheduler 不再自带 WRITE 模式超时自行释放块,块释放统一收敛到 worker ACK 之后。其二,decode 本地 prefix cache 全命中时原会向 prefill 发空 block list,prefill 以 `MoRIIOError("block_notify_list cannot be empty")` 拒绝并扣住块;现在改为直接发 release 消息让 prefill 释放。附带把 notify endpoint 解析(multi-pod host、端口偏移)抽取为 `_resolve_notify_endpoint()` 供正常与失败路径复用。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py: fail_unallocated_write()/_notify_write_failed() 广播 write_failed
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py: get_transfer_results() 汇报 failed_recving;全前缀命中改发 release
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py: WriteTask 增加 remote_dp_size;defer_timeout 语义更新
tests/v1/kv_connector/unit/test_moriio_write_done_routing.py: 3 个新单测(超时广播/ACK 等待/失败上报)
```

核心超时处理:

```python
elif (now - task.enqueue_time > defer_timeout
      and self.worker.moriio_wrapper.fail_unallocated_write(task.transfer_id)):
    self._notify_write_failed(task, now - task.enqueue_time)
...
for decode_dp_rank in range(max(task.remote_dp_size, 1)):
    self.worker.moriio_wrapper.send_notify(
        task.transfer_id, remote_ip, remote_port,
        message_type="write_failed")
```

## 性能声明

无吞吐/延迟提升声明;性能收益体现为消除挂死。PR 正文 E2E 复现(2× Radeon AI Pro R9700S,Llama-3.1-8B-Instruct,1024 in/1024 out,128 请求、并发 64,decode `--max-num-seqs 20`,xgmi backend,叠 #53250):修复后 benchmark 可正常跑完,双方 KV cache 用量归零。`kv_load_failure_policy=fail`(默认)时成功 109、失败 19、deferred write 超时 19;`recompute` 时成功 128、失败 0、超时 34。单元测试 207 个在 ROCm + mori 环境全过(main 为 204 + 新增 3)。

## 验证与回归风险

新增 CPU-only 单测 `test_moriio_write_done_routing.py` 三例:deferred write 过期后 transfer 标记完成、prefill 块释放、`write_failed` 跨 pod 抵达每个 decode DP rank 且迟到分配被忽略;scheduler 等待 worker ACK 而非自行超时释放;归属 rank 将 `write_failed` 上报为 failed_recving。回归风险:WRITE 模式下 scheduler 侧超时收割逻辑被禁用(`mode == MoRIIOMode.READ` 才过期),若 worker ACK 丢失仍有新的挂死窗口(与 #58962 的 prefill 死亡场景互补);`write_failed` 广播量与 decode DP size 成正比;未匹配失败队列截断(4096)意味着极极端积压下失败可能被丢弃。影响面限于 MoRI-IO connector(ROCm/mori 环境)的 WRITE 模式与全前缀命中路径。

## 关联

强依赖: #53250(decode 抢占导致的等待第二个 push,本 PR 靠其完整修复 hang);互补: #58962(decode 侧接收超时,覆盖 prefill 半途死亡场景)。前序: #51681(write_done routing)。本批次内无其他 MoRI-IO 卡。
