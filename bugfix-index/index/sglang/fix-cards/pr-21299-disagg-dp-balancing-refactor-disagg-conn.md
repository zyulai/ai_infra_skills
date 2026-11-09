# PR #21299: [Disagg DP-Balancing] Refactor Disagg Conn and Fix Hang with total_request/total_tokens Balancing

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21299](https://github.com/sgl-project/sglang/pull/21299) |
| 作者 | weireweire |
| 合并日期 | 2026-03-31 |
| labels | high priority, run-ci |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论
PD 分离下 prefill 用 `total_requests`/`total_tokens` 内部 DP 均衡时,decode 各 TP/CP rank 独立解析 pending 请求导致队列状态不一致而 hang;重构 receiver 生命周期:先全 rank 立即入队、DP rank 可用后再 init,元数据发送拆为独立步骤。

## 问题现象
正文披露:prefill 侧使用 `total_requests` 等内部 DP 均衡时,decode 可能在 `bootstrap_room -> prefill_dp_rank` 映射尚未就绪时就收到请求。此前每个 decode TP/CP rank 各自独立解析 pending 请求,可能造成部分 rank 先把请求从 `pending_reqs` 移入 handshake 队列;一旦发生,后续 decode 侧的集合(collective)轮询会看到各 rank 本地队列状态不一致,请求管线停摆(PD disaggregation hang,issue #21297)。

## 根因
正文披露:根因是 receiver 的创建/入队时机与 prefill DP rank 解析耦合——receiver 在 `add()` 时才创建,且创建即需要 `prefill_dp_rank`,各 rank 解析进度不同步导致队列状态跨 rank 不一致、collective polling 卡死。修复思路(按 issue #21680 讨论的新方案,替换此前「单 leader rank 解析 + 广播 + lockstep 入队」的 workaround):receiver 的构造与 bootstrap 初始化解耦,所有 decode TP/CP rank 在 `add()` 时立即创建并入队 receiver(保证队列状态一致),把依赖 bootstrap 的工作延后到 `prefill_dp_rank` 可用时的 `init()`,元数据发送再拆为独立的 `send_metadata()` 步骤。

## 修复方案
1. `disaggregation/base/conn.py`:抽象接口拆分——`init(prefill_dp_rank)`(解析 bootstrap 元数据、标记可传元数据)与 `send_metadata(kv_indices, aux_index, state_indices)`(通知 prefill server)分离;
2. `disaggregation/decode.py` 的 `DecodePreallocQueue.add()`:先 `_create_receiver_and_enqueue(req)` 立即入队(fake transfer 直接 `init(0)`),缓存命中能解析则 `kv_receiver.init(prefill_dp_rank)`,否则进 `pending_reqs`(类型改为 `List[DecodeRequest]`);`_resolve_pending_reqs` 解析后调用 `kv_receiver.init(...)`;`pop_preallocated` 改调 `send_metadata`;
3. `common/conn.py`:`CommonKVReceiver.__init__` 不再接收 `prefill_dp_rank`,初始化 room tracker 与 Bootstrapping 状态,`init()` 中完成 bootstrap info 解析、失败时记录 `conclude_state=Failed` 并标记 KVPoll.Failed,成功后置 WaitingForInput,移除「prefill_dp_rank must be resolved before creating receiver」断言;
4. `fake/conn.py`、`mooncake/conn.py`、`mori/conn.py`、`nixl/conn.py` 各后端同步适配三态 poll(Bootstrapping/WaitingForInput/Success);
5. 新增 `TestDisaggregationDPAttentionTotalRequests`、`TestDisaggregationDPAttentionTotalTokens` 两个 bench_serving 测试(256 prompts,random 2048/512,max_concurrency 128,断言 completed==256)。

关键 diff 摘录:

```diff
+    def init(self, prefill_dp_rank: int):
+        """
+        Resolve bootstrap metadata and mark the receiver ready for transfer metadata.
+        """
+        ...
+
+    @abstractmethod
+    def send_metadata(
         self,
         kv_indices: npt.NDArray[np.int32],
```

正文称用 1p1d disagg DSR1 + GPQA benchmark 验证通过。

## 影响与回归风险
- 回归面:PD 分离 decode 侧全部传输后端(fake/common/mooncake/mori/nixl)的 receiver 生命周期与状态机;`init`/`send_metadata` 接口签名变化影响所有自定义 KVReceiver 实现;`pending_reqs` 语义由 `Req` 变为 `DecodeRequest`;nixl `poll` 在未开始传输时从固定返回 `WaitingForInput` 改为返回当前 `status`。早前 decode-side pending-sync/global-ready 快路径 workaround 被回退。若某后端未正确适配三态轮询,可能出现新的握手卡点(正文以 GPQA 基准与新增双测试覆盖)。

## 关联
- https://github.com/sgl-project/sglang/issues/21297 (修复的 hang issue)
- https://github.com/sgl-project/sglang/issues/21680 (方案讨论)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
