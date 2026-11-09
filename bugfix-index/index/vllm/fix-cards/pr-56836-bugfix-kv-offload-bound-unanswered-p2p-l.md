# PR #56836: [Bugfix][KV Offload] Bound unanswered P2P lookup waits

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56836](https://github.com/vllm-project/vllm/pull/56836) |
| 作者 | jacklin78911-collab |
| 合并日期 | 2026-10-15 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 31 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论

P2P 对端握后不回 LookupMsg 致 RETRY 无限挂起,加 30 秒进度超时降级 MISS。

## 问题现象

P2P KV offload 场景中,握手成功后若对端保持传输连接但停止应答 `LookupMsg`,消费端会无限期返回 `RETRY`,请求一直等不到结果;服务端 pending-lookup 超时在对端 serving loop 停止推进时也帮不上忙。影响面为使用 P2P KV 卸载且对端停滞的部署(对应 #56823 的未应答 lookup 部分)。

## 根因

按正文披露:lookup 探测此前没有超时机制——client.py 原注释写明 "There is no timeout … an unanswered probe simply stays None until then",未应答的探测保持 None(即 RETRY)直到请求结束;对端只保活不回应时消费端无从恢复。这是调度器观测到的"进度超时"问题,不是按 key 年龄或传输到达时间戳的限制。

## 修复方案

`vllm/v1/kv_offload/tiering/p2p/session/` 三处改动:

- `client.py`:新增 `_LOOKUP_TIMEOUT_S = 30.0` 与 `_lookup_deadlines` 字典;首个未应答 lookup 批次发出时启动 30 秒进度截止;解析出此前未应答探测的响应会重置该请求截止,而重复 poll、额外 key、重复/未知响应都不延长;全部探测完成、抓取到匹配前缀或请求结束则清除截止;未发送的探测等 `ConnectAckMsg` 后才开始计时。新增 `get_expired_lookup()`。
- `session.py`:`poll()` 先排空缓冲回复与已完成的传输,再检查 lookup 过期;过期时打警告(含对端、请求 id、无进度时长),best-effort 发送 `DisconnectMsg` 并将整个对端会话标记死亡。
- `server.py`:`collect_results()` 把 `_pending_store_results` 的排空移到传输 finalization 之后,保证 supplied-but-not-fetched 的 store 结果也及时返回。

会话失败策略:过期后对端整会话判死,manager teardown 把未决请求转为 `MISS` 供本地 prefill 回退,保留失败标记直到请求完成并释放会话资源。wire format 与模型计算不变。

关键 diff 摘录(session.py):

```python
expired_lookup = self._client.get_expired_lookup()
if expired_lookup is not None:
    req_id, elapsed = expired_lookup
    logger.warning("... lookup response timed out ... after %.1fs without progress", ...)
    with contextlib.suppress(Exception):
        self._conn.send({TYPE_KEY: DisconnectMsg.TYPE})
    self._conn.mark_dead()
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:P2P KV offload 的 lookup/load/store 路径。健康对端行为不变(响应按期重置截止);停滞对端会话 30 秒后断开、该对端上的其他未决工作走既有断连路径失败;请求级超时改为会话级判死,若误判可能断开仍有产出的慢对端。本 PR 明确不覆盖握手前 `ConnectAck` 等待(由 #51504 负责)。

## 关联

- [issue #56823](https://github.com/vllm-project/vllm/issues/56823)(本 PR 修复其未应答 lookup 部分,issue 需保持打开)
- [issue #51504](https://github.com/vllm-project/vllm/issues/51504)(握手 deadline 路径,仍开放)
- [PR #55962](https://github.com/vllm-project/vllm/pull/55962)(相关但不重叠)
- [PR #53087](https://github.com/vllm-project/vllm/pull/53087)(HIT_PENDING 等待,不同状态)
- [PR #53330](https://github.com/vllm-project/vllm/pull/53330)(供需防护,无 lookup deadline)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
