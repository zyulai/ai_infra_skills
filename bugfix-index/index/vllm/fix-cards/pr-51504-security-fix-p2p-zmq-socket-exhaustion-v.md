# PR #51504: [Security] Fix P2P ZMQ socket exhaustion via unbounded peer sessions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51504](https://github.com/vllm-project/vllm/pull/51504) |
| 作者 | jperezdealgaba |
| 合并日期 | 2026-10-11 |
| labels | documentation, ready |
| 热度 | 👍 2 · 💬 20 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
API 客户端可经 kv_transfer_params 无限提交 peer 地址,每地址 3 个 ZMQ socket 从不回收,约 340 个即耗尽 1023 上限崩掉 EngineCore;加五层防护。

## 问题现象
修复 GHSA-frvg-qw88-hww9:启用 P2P secondary tier(OffloadingConnector + TieringOffloadingSpec + p2p)的部署会从 API 请求的 `kv_transfer_params` 接受 peer 地址。恶意/异常客户端提交大量唯一的非监听 peer 地址,每个地址创建 3 个 ZMQ socket(DEALER + monitor + PAIR)且从不回收;约 340 个这样的 peer 就耗尽 ZMQ context 的 `MAX_SOCKETS=1023` 限额,导致 EngineCore 崩溃。

## 根因
正文披露:P2P 会话的创建完全由客户端提供的 `kv_transfer_params` 驱动,没有任何会话数量上限、握手超时或空闲回收机制;失败/不响应的 peer 的 socket 永远不会被清理,socket 配额随之耗尽。

## 修复方案
五层 defense-in-depth:
- `envs.py` 新增 `VLLM_P2P_MAX_PEERS`(默认 128)、`VLLM_P2P_HANDSHAKE_TIMEOUT_S`(默认 30)、`VLLM_P2P_IDLE_TIMEOUT_S`(默认 300)。
- `manager.py`:`_get_or_create_session` 强制会话上限并捕获 transport 错误(返回 None 而非向上抛);`_accept_new_peers` 在满容量时拒绝入站连接;`on_new_request` 对 None 记录失败请求;新增 `_reap_idle_sessions()`(无 pending work 且超过 idle 超时才驱逐,且仅当剩余槽位距上限不足 `_IDLE_REAP_HEADROOM=8` 时运行,避免拆散安静但健康的 mesh),挂到 `_poll_once()`。
- `session/session.py`:记录 `_connected_at`/`_last_activity_at`;`poll()` 中对已连接但超过握手期限仍未 ready 的连接 `mark_dead`(先排空 inbox,让 deadline 时已缓冲的 ConnectAck 完成握手);收发消息时刷新活动时间戳。
- `control/zmq.py`:monitor 订阅扩展 `EVENT_CLOSED`、`EVENT_HANDSHAKE_FAILED_*` 等致命事件;`_check_monitors()` 改为每次排空全部 pending 事件并处理所有致命事件。
- `docs/usage/security.md` 新增 "P2P KV-Offload Peer Session Limits" 小节(三个环境变量与部署建议,如对外部署调低 MAX_PEERS、反代剥离 kv_transfer_params)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;暴露面是开启 P2P KV-offload 且 API 可达的部署。
- 回归面:P2P tiering 会话生命周期——超过 128 个并发 peer 的合法部署会拒绝新连接(可调 env);握手超过 30s 的慢 peer 会被拆掉(缓冲 ACK 场景已处理);空闲会话在接近容量上限时被驱逐、重连需重新握手;transport 失败从崩溃改为该请求回落 local prefill。195 个既有+新增 P2P 测试通过(76 manager + 100 session + 19 zmq transport)。

## 关联
- GHSA-frvg-qw88-hww9(正文提及的安全通告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
