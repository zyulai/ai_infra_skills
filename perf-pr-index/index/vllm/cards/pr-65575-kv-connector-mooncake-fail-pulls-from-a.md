# PR #65575: [KV Connector][Mooncake] Fail pulls from a lost prefill worker

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @CJJ1008 |
| 合并 | 2026-11-16 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | documentation, ready, kv-connector, mooncake |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py、vllm/utils/network_utils.py、docs/features/mooncake_connector_usage.md、tests/v1/kv_connector/unit/test_mooncake_connector.py、tests/v1/kv_connector/unit/test_mooncake_stats.py |
| 链接 | https://github.com/vllm-project/vllm/pull/65575 |

## 一句话结论

Mooncake decode 端为每个 pull socket 挂 ZMQ 连接监控并启用 ZMTP 心跳,prefill worker 进程死亡后等待中的请求在 2–4 秒内失败(此前要等 545 秒再加一波 540 秒),并清掉死地址缓存让后续请求重新查询 bootstrap server。

## 动机

prefill(P) worker 死亡时 decode(D)无感知:D 的 DEALER 会静默重连或对拒绝连接持续重试,pull 只能等到 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60` = 540 s 超时,期间一直占着请求的 KV blocks;且 reply 无法到达——DEALER 没有 routing id,P 的 ROUTER 会丢弃发往已失联连接的 reply。同时 `_remote_agents` 从不失效,排在其后的请求继续向死地址发 pull、再等一个 540 s,负载下 D 的 KV 被永远完不成的请求塞满。#35943 与 #37745 曾报告此 hang,均以 not planned 关闭。

## 优化点

1. D 监控每个 pull socket 的 `EVENT_DISCONNECTED | EVENT_CONNECT_RETRIED`(`_lost_peer_monitor` + `_recv_unless_lost` 用 `asyncio.wait(FIRST_COMPLETED)` 竞速 recv 与 monitor):任一事件即经 `_handle_failed_recv` 失败该 pull,并删除列出该 worker 的所有 engine 的缓存地址;后续请求重新查 bootstrap——死 P 立即失败,重启的 P 能被重新找到。
2. ZMTP 心跳(`_PULL_HEARTBEAT_IVL_MS` 2000 / `_PULL_HEARTBEAT_TIMEOUT_MS` 10000,与 `kv_offload/tiering/p2p/control/zmq.py` 一致)让冻结或不可达的 P 也会掉线;libzmq 只在 `connect()` 前设置的选项生效,故 `make_zmq_socket` 新增 `heartbeat_ivl_ms` / `heartbeat_timeout_ms` 参数。
3. 失败沿既有 `kv_load_failure_policy` 处理,对活着的 P 无行为变化、无新选项。

## 关键改动

```python
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
+_PULL_LOST_EVENTS: Final[int] = zmq.EVENT_DISCONNECTED | zmq.EVENT_CONNECT_RETRIED
+_PULL_HEARTBEAT_IVL_MS: Final[int] = 2000
+_PULL_HEARTBEAT_TIMEOUT_MS: Final[int] = 10000
```

```python
+                        ret_msg = await _recv_unless_lost(sock, monitor)
+                        if ret_msg is None:
+                            # Look P up again: it may have restarted elsewhere.
+                            for engine_id, agents in list(self._remote_agents.items()):
+                                if any(worker_addr in pp_addrs.values()
+                                       for pp_addrs in agents.values()):
+                                    del self._remote_agents[engine_id]
+                            self._handle_failed_recv(
+                                pull_metas, req_ids,
+                                f"lost connection to prefill worker {worker_addr}")
+                            return
```

## 性能声明

数字均照抄 PR 正文(未独立核实):

- P 进程死亡:等待中的请求在 2–4 s 内失败,而不是 545 s 之后再叠加一波 540 s。
- P 冻结或不可达:在 heartbeats 或既有超时到期后失败,全部在 102 s 内(冻结 P:先等 ZMQ 30 s handshake 超时,其后排队的请求等 bootstrap 查询超时 3 × `VLLM_MOONCAKE_CONNECTOR_TIMEOUT`)。
- 成本:每个 pull 一个 inproc monitor socket,约 30 µs;对活 P 无变化。
- 端到端(2× A800,DeepSeek-MoE-16B,TP=1 每侧,`kv_lease_duration=30`;128 请求 ≈2.4× D 的 KV 容量;P 进程组在 t≈41 s 收到信号,时间从信号起算):main SIGKILL:stranded requests fail 545 s, then 1,084 s;D KV 低于 50% 用 556 s;ok/failed 53/75。本 PR SIGKILL:2–4 s;5.5 s;54/74。本 PR SIGSTOP:12–102 s;109 s;53/75。

## 验证与回归风险

- 单测:`pytest tests/v1/kv_connector/unit/test_mooncake*.py` → 750 passed, 5 skipped(另有 4 个用例因 HF Hub 拉取错误重跑通过)。新测试在子进程里跑一个"收下 pull 但永不回复"的 ZMQ ROUTER,随后 kill / `SIGSTOP` / 从未监听三种情形:main 上全部超时,本 PR 约 1 s 内通过。顺带修复自 #64193 起坏掉的 `test_mooncake_stats.py::test_expired_request_bumps_counter`。pre-commit 通过。
- 回归风险/tradeoff:连接遇到瞬时网络错误的 pull 现在会直接失败,而 ZMQ 原本会重试;冻结 P 时在 D 察觉前发出的 pull 仍要等 30 s handshake 超时;监控 socket 是每 pull 一个的常驻成本。

## 关联

是 #64193(P 侧)的 decoder 侧对应物;#35943、#37745 为关闭的历史报告;文档 `mooncake_connector_usage.md` 同步更新了 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 语义。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
