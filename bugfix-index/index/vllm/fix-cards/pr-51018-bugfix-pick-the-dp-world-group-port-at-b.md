# PR #51018: [BugFix] Pick the DP world-group port at bind time via the coordination store

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51018](https://github.com/vllm-project/vllm/pull/51018) |
| 作者 | aoshen02 |
| 合并日期 | 2026-10-08 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 25 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
DP world group 预分配端口在 bind 前被抢占致启动挂死;改由 rank 0 bind 时经 coordination store 发布端口。

## 问题现象
`--data-parallel-size N` 下,DP world group 的 rendezvous 端口来自 `_data_parallel_master_port_list`:config 时探测一次,而引擎在数分钟后(worker spawn、CUDA init、model load 之后)才在 `init_process_group` 使用,期间无人持有端口;同机其他进程——最常见是冷启动的兄弟 vLLM 实例(RL rollout fleets 标准形态)——可将其抢占,启动在最后一步失败并挂起:
```
torch.distributed.DistNetworkError: The server socket has failed to
listen on any local network address. port: <n>, ... EADDRINUSE
```
服务器不退出,而是模型加载数分钟后挂住。A/B(GB200 单节点,squatter 抢占 `[VLLM_PORT, VLLM_PORT+16)`):main 6/6 启动失败,修复后 8/8 健康;两实例并发冷启动 main 6/6 失败,修复后 6/6 健康;nightly 上确定性复现同样结果。

## 根因
正文披露:预分配端口「探测时选定」与「实际 bind」之间存在竞争窗口(probe-to-bind race);#36330 已为 elastic-EP stateless groups 修复同类缺陷(rank 0 bind 临时端口、保留监听 socket、经 coordination TCPStore 发布端口),本 PR 把该机制扩展到最后一个仍预分配端口的消费者——world group。

## 修复方案
- `vllm/v1/engine/utils.py` `launch_core_engines()`:无既有 store 时为 online DP 创建 coordination TCPStore(条件:`dp_size > 1`、非 offline、`dp_rank == 0` 且 `local_engine_count > 0`、非 elastic EP;后续提交移除最初错误的 MoE-only gate,否则非 MoE 复制式 DP 仍走旧竞争路径),端口经既有启动握手字段 `_coord_store_port` 下发(该字段加入握手拷贝列表)。
- `vllm/distributed/parallel_state.py` `init_distributed_environment()`:DP 路径下 group rank 0 在 port 0 创建 world-group TCPStore——内核分配端口且 store 从诞生起持有它——再经 coordination store 发布端口;其他 rank 阻塞读取并连接;所有 rank 把 store 以 `store=` 传给 `init_process_group`,端口在选定与使用之间从不释放(torch 要求 store 与 init_method 互斥,故 `init_method=None if world_pg_store else ...`);无 coord store(offline、external launcher/SPMD、elastic、版本 skew)时逐字节回退旧行为。
- 另修 MI355 CI 暴露的 fork 问题:`get_cached_tcp_store_client` 每进程缓存一个已连 client,fork 后同一 engine 的两个 TP sibling 复用同一 socket 交织请求;port 交换改为每 worker 开一次性短命 client,缓存 client 的 fork 不安全留给后续处理。

## 影响与回归风险
- 回归面:online DP(mp/ray)引擎启动路径与 `init_distributed_environment` 的 DP 分支、fork worker 的 store client 使用;offline/SPMD/elastic-EP/单引擎部署回退不变;rank 0 以 `wait_for_workers=False` 建 store,缺 worker 的失败从 store 构造超时移到 PG-init rendezvous 超时(与 torchelastic 部署一致)。

## 关联
#36330(机制来源);症状报告 #39015、#28498、#14919

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
