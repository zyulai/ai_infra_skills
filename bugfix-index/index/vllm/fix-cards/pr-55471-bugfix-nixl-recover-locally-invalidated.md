# PR #55471: [Bugfix][NIXL] Recover locally invalidated pull peer metadata

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55471](https://github.com/vllm-project/vllm/pull/55471) |
| 作者 | xijiaat |
| 合并日期 | 2026-10-06 |
| labels | bug, ready, kv-connector |
| 热度 | 👍 0 · 💬 34 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
Prefill 引擎 ID 不变但 Decode 侧 NIXL 本地 peer 元数据失效时,缓存的 `_remote_agents` 与已准备 dlist 让后续请求一直走陈旧路径;检测确认缺失后清理并重新握手恢复。

## 问题现象
在 P/D 分离(拉模式)下,Decode 端的本地 NIXL remote 元数据被本地失效(如受控实验中仅移除 Decode 的 local remote metadata),而 Prefill 进程仍存活且 engine ID 不变。此时后续请求继续使用缓存的 `_remote_agents` 条目和已准备的 descriptor list,真实读取以 `NIXL_ERR_NOT_FOUND` 失败(stale read)。与 #50047 的场景区分:那个是"同一地址确认了新 engine ID"需要替换 peer,本 PR 是"engine ID 不变但 NIXL 丢失了解码端本地 peer 元数据"需要恢复。

## 根因
正文披露:无恢复机制时,vLLM 缓存的 `_remote_agents` 条目与已准备的 dlist 会把后续请求持续留在陈旧路径(stale path)上——即没有任何逻辑在"读取失败"后检查本地元数据是否已不存在并清理重建。已合并的 #50047 只覆盖 engine ID 变更(替换)场景,未覆盖同 ID 元数据失效场景。

## 修复方案
生产改动在 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` 与 `pull_worker.py`:
1. `NixlBaseConnectorWorker` 新增 `_failed_remote_engines`、`_invalid_remote_engines` 集合与 `_invalid_engine_backoff` 字典;`_handle_failed_transfer` 在 pull 模式且请求带 remote 元数据时把 engine_id 记入 failed 集合。
2. 原 `_cleanup_replaced_remote_engines` 泛化为 `_cleanup_remote_engines`,一个 pass 同时处理替换与失效:对 failed peer 用无 descriptor 的 `check_remote_metadata(agent)` 探测——该查询只看**本地元数据是否存在,而非 peer 健康**;只有确认缺失才标记可恢复,传输错误或探测异常本身不算。等待后台握手完成才探测/清理;失效 peer 拒绝新读取,等所有在途 receive handle 结束后才清理;复用 `_cleanup_remote_engine()` 释放资源,TTL 逐出与停机也会清掉恢复状态。
3. `_mark_remote_engine_invalid` 实现指数退避(1s 起、上限 60s、翻倍):反复失效(如持续性 RDMA 故障)的 peer 在退避期内快速失败而不反复重握手。
4. `pull_worker.py`:`start_load_kv` 去掉两处内联的"engine 不在则握手"分支,统一走 `_read_blocks_for_req`;其中失效 peer 直接按失败处理,engine 不在则触发重新握手。清理后新请求用**不变的 engine ID** 重新握手并使用新准备的 descriptor;失败请求不会用旧 remote block ID 重放。
5. 测试复用 `TestPeerReplacement` 与 `FakeNixlWrapper`,新增确认缺失/健康/探测异常、同 ID 重握手、pending-handshake 门控、handle 排干顺序、新读拒绝、健康 peer 隔离、替换/失效重叠、TTL、停机等用例。

## 影响与回归风险
- 影响窗口:正文未说明;恢复仅限 pull 模式,push 模式留有明确 TODO。
- 回归面:NIXL connector 的 pull 读路径、失败统计(`record_failed_transfer`)与清理逻辑。作者明确声明边界:恢复不代表每个 UCX 失败都可恢复;#52232 的 HMA 失败传播问题不在本 PR 内(仅靠本 PR,恢复前的失败请求仍可能输出无效 token——历史实验中一次失败请求仍发出了 32 个无效 token,需叠加 #52232 与 API 终态错误才全通过);TTL 策略与 NIXL #1987 原生 UCX 竞态也超出范围。验证:Buildkite #91865 NVIDIA/AMD KV Connector shards 4/4 通过;本地 32 个用例;H20 上真实 NIXL/UCX probe 字节校验读取通过。

## 关联
- #50047(follow-up)
- #49238(related)
- #52232(HMA 失败传播,不在本 PR)
- NIXL #1987(native UCX 竞态,超出范围)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
