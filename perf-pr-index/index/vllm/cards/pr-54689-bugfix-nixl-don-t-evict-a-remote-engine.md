# PR #54689: [Bugfix][NIXL] Don't evict a remote engine a transfer is still reading from

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jyizheng |
| 合并 | 2026-09-12 |
| 分类 | Overlap / Scheduler |
| Label | bug, ready, kv-connector |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/54689 |

## 一句话结论

修复 NIXL PD 分离下在途 KV transfer 的远端 engine 被 TTL 误驱逐、导致 decode worker 无声崩溃的问题。

## 动机

NIXL connector 的 `_evict_stale_engines` 依赖 `_engine_last_active` 判断远端 engine 是否 stale,但该时间戳只在 `pull_worker._read_blocks_for_req` 发起读时打一次,读取进行期间不再刷新。一个存活超过 `engine_ttl`(默认 3600 s)的在途 transfer——对端丢失 NIC 时会无限期挂住——会让其 engine 看起来 idle 而被驱逐;驱逐会对其正在使用的状态调用 `release_dlist_handle` 与 `remove_remote_agent`,释放发生在 NIXL 内部而非 Python 层,症状是 worker 进程无 traceback 直接死亡。实际观测来自一个 10-prefill/3-decode 的分离部署:三个 decode worker 在一次新远端握手 6 秒后(恰为驱逐运行时机)同时死亡,此前已空闲三小时、积累了一批 TTL 过期 engine。

## 优化点

驱逐逻辑现在先收集仍在被读取的远端 engine 集合:新增 `_engines_with_inflight_transfers()`,遍历 `_recving_transfers` 并从 `_recving_metadata` 取出每个在途请求的 `meta.remote.engine_id`;`_evict_stale_engines` 在 TTL 判定之外增加 `eid not in busy` 条件,跳过这些 engine。transfer 结束或失败后 engine 立即恢复可驱逐状态,因此对真正 idle 的 engine,TTL 本要约束的内存上界行为不变。文档字符串同步更正,明确"in-flight transfer 显式 hold 住其 engine"。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py: 新增 _engines_with_inflight_transfers,驱逐时跳过在途 engine
tests/v1/kv_connector/unit/test_nixl_connector.py: 新增 test_engine_with_inflight_transfer_is_not_evicted
```

```python
busy = self._engines_with_inflight_transfers()
for eid, last_active in list(self._engine_last_active.items()):
    if now - last_active > self._engine_ttl and eid not in busy:
        self._cleanup_remote_engine(eid)
```

## 性能声明

PR 未提供 benchmark 数字。此为可靠性修复:保证长时运行的 PD 分离 KV 传输( disaggregated serving)不被 TTL 驱逐打断,间接支撑该部署架构的持续吞吐。

## 验证与回归风险

新增单测将 engine 老化超过 TTL 并挂一个在途 transfer,断言 `release_dlist_handle`/`remove_remote_agent` 均不被调用、engine 仍注册;删除 transfer 后再次驱逐,engine 正常回收。影响面限于 NIXL KV connector 的 TTL 驱逐路径;对真正空闲的 engine 行为不变,不引入额外的内存驻留风险。

## 关联

同区域(kv-connector)可靠性修复:PR #55053(MoRI-IO speculative decoding 下 KV block-offset 修复)。无其他直接关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
