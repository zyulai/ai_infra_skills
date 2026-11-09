# PR #54689: [Bugfix][NIXL] Don't evict a remote engine a transfer is still reading from

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54689](https://github.com/vllm-project/vllm/pull/54689) |
| 作者 | jyizheng |
| 合并日期 | 2026-09-12 |
| labels | bug, ready, kv-connector |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
仍在进行中的传输不会刷新引擎活跃时间戳,TTL 驱逐会释放其正在读取的远端资源导致 worker 无栈崩溃;驱逐时跳过有 in-flight 传输的引擎。

## 问题现象
NIXL KV connector 的 pull worker 进程无 traceback 直接死亡:`Worker proc VllmWorker-0 died unexpectedly (exit code: None), shutting down executor`。实际观察:10 prefill / 3 decode 的 PD 分离部署中,三个 decode worker 在几秒内相继死亡,时间点在一次 new-remote 握手后 6 秒(正是驱逐运行时机),此前引擎已闲置 3 小时,积累了一批超过 TTL 的引擎。

## 根因
正文披露:`_evict_stale_engines` 依赖的不变量是「有活跃传输或未完成握手的引擎不可能是 stale:活跃传输会在 start_load_kv 中触碰 _engine_last_active」。但该时间戳只在读发起时由 `pull_worker._read_blocks_for_req` 盖一次,读进行期间不再刷新。于是一个存活超过 engine_ttl(默认 3600 秒;丢失 NIC 的对端会无限期持有传输)的传输会让其引擎看起来处于空闲状态。驱逐随即对该传输正在读取的状态调用 `release_dlist_handle` 和 `remove_remote_agent`;没有任何代码查询 `_recving_transfers`,且释放在 NIXL 内部(Python 之外)发生,所以表现为 worker 无栈死亡。

## 修复方案
修改 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`:
1. 新增 `_engines_with_inflight_transfers()`,从 `_recving_transfers` / `_recving_metadata` 收集仍有传输在读取的远端 engine_id 集合;
2. `_evict_stale_engines` 中,超过 TTL 且不在该集合中的引擎才执行 `_cleanup_remote_engine(eid)`;
3. 更新 docstring 并新增测试 `test_engine_with_inflight_transfer_is_not_evicted`(TTL 过期 + in-flight 传输时不释放;传输结束后可正常驱逐)。

关键 diff 摘录:
```python
busy = self._engines_with_inflight_transfers()
for eid, last_active in list(self._engine_last_active.items()):
    if now - last_active > self._engine_ttl and eid not in busy:
        self._cleanup_remote_engine(eid)
```
传输完成或失败后引擎即恢复可驱逐,对真正空闲引擎的内存约束(TTL 的本意)不变。

## 影响与回归风险
- 回归面:仅 NIXL connector 的 stale 引擎驱逐判定;有 in-flight 传输的引擎会被多保留一段时间(至传输结束),其余驱逐行为不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
