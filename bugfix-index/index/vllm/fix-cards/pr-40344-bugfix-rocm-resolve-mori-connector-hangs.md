# PR #40344: [Bugfix][ROCm] Resolve MoRI connector hangs at high concurrency

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40344](https://github.com/vllm-project/vllm/pull/40344) |
| 作者 | simondanielsson |
| 合并日期 | 2026-05-28 |
| labels | bug, kv-connector, ready, rocm, v1 |
| 热度 | 👍 2 · 💬 19 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论

MoRI-IO connector 在高并发下多处缺陷导致无限挂起;禁用带内通知、处理失败传输、补全超时与映射修复,512 并发可用。

## 问题现象

使用 MoRI-IO KV connector 做 1P1D 分离部署时,高并发下 connector 无限挂起(对应 issue #40340)。main 分支还会直接崩溃:`/app/mori/src/io/rdma/backend_impl.cpp:291 ... Assertion 'msg.totalNum > 0' failed.`。修复后 READ 与 WRITE 模式均可跑至少 512 并发:实测 DeepSeek-R1-0528(TP=8、fp8 KV cache)1p1d 部署,256 并发 2558/2560 成功、512 并发 5114/5120 成功;GSM8k exact_match 0.9439(flexible)/0.9378(strict)。

## 根因

正文逐条披露了导致挂起的多个缺陷:

1. MoRI 带内通知在高并发下耗尽 QP send queue,毒化 transfer 状态,请求卡死在 `WAITING_FOR_REMOTE_KVS`(而 vLLM 本来就用 ZMQ 做完成通知);
2. `_pop_done_transfers` 不处理 `Failed()` 传输,失败传输永远滞留 `_recving_transfers`,prefill 侧块不释放;
3. producer 侧 `transfer_id<->request_id` 映射过早移除,没有请求被标记为 done sending,`get_finished` 不释放(仅靠 `update_connector_output` 兜底);
4. 完成通知丢失(如 `ibv_post_send` 失败)时 KV 块永不释放;
5. WRITE 模式存在竞态:请求加入 `transfer_id_to_request_id` 与 `pop_finished_write_req_ids` 弹出的时机不匹配;
6. READ 模式下 D 侧请求被立即调度为 RUNNING(绝不进入 WAITING_FOR_REMOTE_KVS),却上报 `done_recving`,触发断言错误;
7. `status.Wait()` 是无限忙等自旋;
8. `while True` 忙等循环无让出;
9. 代码中 request id 与 transfer id 使用多处错位。

## 修复方案

改 `moriio_common.py`、`moriio_connector.py`、`moriio_engine.py`:RDMA 后端设 `enable_notification=False`(改用 ZMQ 通知);`_pop_done_transfers` 处理 `Failed()` 分支——发通知让 prefill 释放块、不加入 done_req_ids(请求走超时中止);producer 侧保留映射直到传输确认完成(`update_connector_output` 中异步 unmap);新增 `_deferred_send_deadlines`,超时(defer_timeout,默认 60s)未收到 finished_sending 的延迟发送被强制注入 `finished_sending` 释放块;WRITE 模式消费端用 `_unmatched_write_completions` 累积未匹配的完成事件逐步重试;READ 模式不再上报 `done_recving`(仅清理内部状态并发 notify);两处 `while True` 忙等改为带 transfer_timeout(默认 30s)的轮询加 1ms sleep;`waiting_for_transfer_complete` 由 `status.Wait()` 无限忙等改为带截止时间的 Succeeded/Failed/超时轮询,超时或失败汇总抛 `TransferError`;`_finalize_if_complete` 与 `_mark_request_done` 统一用 transfer_id 标记完成。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;影响使用 MoRIIOConnector 的 ROCm 1P1D 分离部署。
- 回归面:改动全部位于 MoRI-IO connector 路径。行为变化:传输失败/超时现在会抛 `TransferError`(此前可能无限等)或强制释放块,而非挂起;READ 模式不再产生 `done_recving` 信号,依赖该信号的调度逻辑需与 scheduler 预期一致;禁用带内通知后,若某路径曾依赖 `PopInboundTransferStatus` 获取状态,需确认 ZMQ 通知覆盖;defer/transfer 超时默认值(60s/30s)可经 `kv_connector_extra_config` 调整,极端慢网络下过短的超时可能误杀正常传输。测试为 256/512 并发 bench 与 GSM8k,2-6 个请求失败。

## 关联

- #40340(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
