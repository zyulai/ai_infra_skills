# PR #18485: [Core] Fix abrupt request abort

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18485](https://github.com/vllm-project/vllm/pull/18485) |
| 作者 | NickLucche |
| 合并日期 | 2025-06-06 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 16 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
KV Connector(P/D)下请求在调度前被 abort 时,`_connector_finished` 查询未注册请求的 block ids 断言崩溃;改为查不到时返回空 block 列表。

## 问题现象
使用 KV Connector(如 NixlConnector `kv_role: kv_both` 的 P/D 分离部署)时,请求在被调度之前(即 kv cache manager 尚未为其分配任何 block)被客户端 abort,decoder 端 EngineCore 直接 fatal error 崩溃,客户端收到 500。错误栈末尾:
```
File "vllm/v1/core/sched/scheduler.py", line 952, in _connector_finished
    block_ids = self.kv_cache_manager.get_block_ids(request.request_id)[0]
File "vllm/v1/core/kv_cache_manager.py", line 369, in get_block_ids
    assert request_id in self.single_type_manager.req_to_blocks
AssertionError
```
正文说明该问题依赖时序、不保证 100% 复现,并给出可稳定复现的 P/D 双实例 + toy_proxy_server + `simulate_abort.py`(10 个请求、abort 第 1、9 个)配置。

## 根因
正文披露:`_connector_finished` 在请求结束时被调用,无条件调 `kv_cache_manager.get_block_ids(request_id)`;对调度前就被 abort 的请求,kv cache manager 的 `req_to_blocks` 中没有该 request_id,断言失败。作者指出这种情况下仍应调用 connector(让它做外部清理),只是应传空 blocks。

## 修复方案
- `vllm/v1/core/kv_cache_coordinator.py` 的 `get_blocks()`:由 `manager.req_to_blocks[request_id]` 改为 `manager.req_to_blocks.get(request_id) or []`,未知请求返回空列表,不再触发断言。
- `vllm/v1/core/sched/scheduler.py`:`_connector_finished` 改为 `(block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)` 解包;"KV connector 只支持一个 KV cache group" 的断言从每次调用上移到 `Scheduler.__init__` 创建 connector 时一次性检查;`_update_waiting_for_remote_kv` 同步该重构并补 `assert self.connector is not None`。

## 影响与回归风险
- 回归面:仅影响配置了 KV Connector 的部署(abort 未调度请求的路径);无 connector 时 `_connector_finished` 提前返回,行为不变。单 KV cache group 限制由运行时报错变为启动时报错,更早暴露。空 block 列表会传给 `connector.request_finished`,connector 实现需容忍空列表(diff 内未展示 connector 侧改动)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
