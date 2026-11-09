# PR #28897: [PD] Fix data race in NixlKVManager for NIXL backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28897](https://github.com/sgl-project/sglang/pull/28897) |
| 作者 | nvjullin |
| 合并日期 | 2026-06-24 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 26 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
NIXL agent 两个线程在 syncMode=none 下竞争 remoteSections_ 哈希表致 segfault;强制 STRICT 同步模式修复。

## 问题现象
在大规模 disaggregated(PD 分离)部署下偶发 segfault,作者注明"hard to hit unless you have a large disagg setup"。崩溃堆栈落在 `nixlAgent::getXferStatus(nixlXferReqH*)` 内部对 NIXL `remoteSections_`(`std::unordered_map<std::string, nixlRemoteSection>`)的 `std::_Hashtable::find` 调用上,顶层表现为 "!!!!!!! Segfault encountered !!!!!!!"。

## 根因
正文披露:`NixlKVManager.__init__` 在 PREFILL 模式下为每个 transfer_queue 启动 `transfer_worker` 线程(内部运行 `self.agent.check_xfer_state`),又通过 `_start_bootstrap_thread` 启动 bootstrap 线程(内部运行 `self.agent.add_remote_agent`)。这两个函数在释放 GIL 的状态下运行,并发竞争 NIXL `src/core/agent_data.h` 中的 `remoteSections_` 哈希表。NIXL 虽有保护该表的锁,但 `nixlAgentData` 构造时是否真正加锁由 agent 配置的 `syncMode` 决定(`lock(effectiveSyncMode(config.syncMode, needsCommThread_))`);SGLang 当前的配置把 `syncMode` 设为 none,即不加锁,从而形成数据竞争。

## 修复方案
改 `python/sglang/srt/disaggregation/nixl/conn.py` 的 agent 初始化:创建 `nixl_agent_config` 时不再使用默认(无同步),显式传入 `sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT`,并加注释说明 `transfer_worker` 与 `_start_bootstrap_thread` 并发运行,不能使用线程不安全的 `sync_mode=None`。

关键摘录:
```diff
-        agent_config = nixl_agent_config(backends=[], num_threads=num_threads)
+        # self.transfer_worker and self._start_bootstrap_thread runs concurrently
+        # so we cannot use sync_mode=None which is thread-unsafe.
+        agent_config = nixl_agent_config(
+            backends=[],
+            num_threads=num_threads,
+            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
+        )
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动为单点配置,作用于所有使用 NIXL 后端的 PD disaggregation 部署(prefill 与 decode 两侧的 agent 均以 STRICT 模式创建)。所有跨线程访问 `remoteSections_` 的 NIXL 内部操作将受锁保护,消除竞争;代价是 NIXL 内部同步开销可能增加(正文未提供性能数据);对非 disagg 路径无影响。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
