# PR #55471: [Bugfix][NIXL] Recover locally invalidated pull peer metadata

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xijiaat |
| 合并 | 2026-10-06 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | bug, ready, kv-connector |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/55471 |

## 一句话结论

NIXL pull 模式下 Prefill 端仍存活但 Decode 端本地 peer metadata 丢失时,自动探测、清退并按原 engine ID 重新握手,恢复 P/D 分离的 KV 拉取。

## 动机

P/D 分离部署中,Decode 侧 NIXL wrapper 的本地 peer metadata 可能被失效(如 UCX 底层故障),而 Prefill 进程仍以相同 engine ID 存活。vLLM 缓存的 `_remote_agents` 条目与已准备的 descriptor list(dlist)会让后续请求持续走 stale 路径,读取失败反复出现。已有 #50047 只覆盖 "同地址出现新 engine ID" 的替换场景;本 PR 处理 "engine ID 未变但本地 metadata 丢失" 的场景,二者共用一套清理通路。

## 优化点

机制:(1) 探测判据严格——receive 失败只把 engine 记入 `_failed_remote_engines`,随后调度一次**无 descriptor 的** `check_remote_metadata(agent)` 检查,它查询的是本地 metadata 是否存在而非对端健康度;只有确认缺失才标记 `_invalid_remote_engines`,传输错误或探测异常本身不触发恢复(测试覆盖 healthy/probe-error/confirmed-missing 三分支)。(2) 统一清理——`_cleanup_replaced_remote_engines` 扩展为 `_cleanup_remote_engines`,一次 pass 同时处理替换与本地失效,复用 `_cleanup_remote_engine` 释放 dlist handle、remove_remote_agent 并清理全部 per-engine 状态;replaced 与 invalidated 重叠的 peer 只清理一次且保留替换方的地址映射;TTL 驱逐与 shutdown 同步清空恢复状态。(3) 排空与拒绝——后台握手进行中(可能 GIL 释放地加载 native metadata)推迟探测与清理;invalid peer 拒绝新读取(`_read_blocks_for_req` 快速失败)并等所有在飞 receive handle settle 后才释放;失败请求不会用旧 remote block ID 重放。(4) 恢复路径——清理后新请求经 pull worker 既有读入口对未变的 engine ID 重新握手,使用新准备的 descriptors;重复失效采用指数退避(`_INVALID_PEER_MIN_BACKOFF_S=1s` 起、翻倍、上限 60s,peer 持续健康超上限后遗忘),避免持久 RDMA 故障下每个请求都重握手。保持 pull-only,push 模式留 TODO;成功传输不引入任何探测。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py: _failed/_invalid_remote_engines 跟踪、_cleanup_remote_engines 统一 pass、_mark_remote_engine_invalid 退避
vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py: _read_blocks_for_req 处理 invalid(快速失败)与缺失(重握手);start_load_kv 简化
tests/v1/kv_connector/unit/test_nixl_connector.py: FakeNixlWrapper.check_remote_metadata 与 12 个恢复用例
```

关键摘录(确认缺失才标记):

```python
try:
    # No descriptors: query local existence, not peer health.
    if not all(map(self.nixl_wrapper.check_remote_metadata, agents)):
        self._mark_remote_engine_invalid(engine_id, now)
except Exception:
    logger.warning("Could not check local NIXL metadata for engine %s.",
                   engine_id, exc_info=True)
```

## 性能声明

无吞吐性能声明;本 PR 为可靠性修复。历史实验证据(仅对当时的 revision/config 有效,重构后未重跑):H20 容器 native 探测(NIXL 1.4.1、PyTorch 2.13.0+cu130)通过 1 MiB CPU/GPU buffer 的真实 NIXL/UCX 读写;TP1 P/D serving 实验(DeepSeek-V4-Flash-0731,NIXL 1.3.2,fp8 HMA KV,eager,`UCX_TLS=tcp,cuda_copy`)中受控移除 Decode 本地 metadata 后 stale read 报 `NIXL_ERR_NOT_FOUND`,新请求自动对未变 engine ID 重握手并正常返回。注意:recovery-only diff 下失败请求仍会输出 32 个 invalid token,需叠加 #52232 与 API terminal-error 修复才得到无内容的失败请求。

## 验证与回归风险

测试:本地 32 个用例(20 个恢复 + 12 个既有 receive-lifecycle)通过,且旧 fixture 负控下 12 个用例失败、recorded-main 负控 3 个预期恢复断言失败 + 5 个通过;pre-commit(Ruff/mypy)通过;Buildkite #91865 NVIDIA KV Connector shards 4/4、AMD KV Connector shards 4/4 通过(整体 "passed and blocked",未执行全部 pipeline job)。回归覆盖:确认缺失 vs 健康/探测异常、同 ID 重握手 fresh descriptors、pending-handshake 门控、sibling handle 排空顺序、新读拒绝、健康 peer 隔离、替换/失效重叠、notify-only、过期计账、清理异常、TTL 驱逐、shutdown。局限:本地全量 pytest 因缺 tblib 受阻,32 个本地用例不能替代 CI;重构后无新的 native NIXL/GPU 或 model-serving 评测;卡死读/握手会推迟恢复,不强制释放活跃资源;不声称所有 UCX 故障皆可恢复。

## 关联

Related to #49238;#50047 的后续(共用清理通路);scope 之外:#52232(HMA 失败传播)、NIXL #1987(UCX race)、TTL 策略。同区域为 KV connector(NIXL、OffloadingConnector,如 PR #52771)。 maintainer 简化(Nick Hill)与 base-worker 化(Yannik Hinteregger 建议)已并入并保留署名。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
