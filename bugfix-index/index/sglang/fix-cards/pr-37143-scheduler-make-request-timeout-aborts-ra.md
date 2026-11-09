# PR #37143: [Scheduler] Make request-timeout aborts rank-consistent to fix TP collective hangs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37143](https://github.com/sgl-project/sglang/pull/37143) |
| 作者 | nogumanov |
| 合并日期 | 2026-09-08 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
各 TP rank 独立执行请求超时扫描,时钟偏差致队列分歧、collective 不匹配挂死;改为仅请求拉取 rank 发 abort 并随广播执行。

## 问题现象
启用 `SGLANG_REQ_WAITING_TIMEOUT` / `SGLANG_REQ_RUNNING_TIMEOUT` 且 TP > 1、队列等待超过 timeout(饱和的长上下文流量)时,服务器挂起,直到 detokenizer heartbeat 健康检查把它杀掉。作者在 GLM NVFP4、B200 TP4、60k-300k token prompt、`SGLANG_REQ_WAITING_TIMEOUT=45` 条件下可稳定复现(修复前 16-90 分钟内必现)。per-rank scheduler 追踪显示各 rank 在分歧前 recv/token/kv 状态完全一致:一次运行按 2+2 分裂(两个 rank 准入 chunked prefill、两个留在 decode),另一次 1+3——两种情况都是某个 rank 比其他 rank 早一个 iteration 把超时请求从队列移除;同一镜像上把 timeout 设为 0 即不再挂。修复后重新启用 timeout,21.6 小时饱和长上下文基准 0 次挂起。

## 根因
正文披露:超时由 wall-clock 扫描在每个 TP rank 上独立执行。停在超时边界的请求会在"最先到达 deadline 的那个 rank 的时钟"(亚毫秒级偏差)上被 abort,导致各 rank 的 waiting queue 组成出现分歧,extend-vs-decode 准入决策分裂,各 rank 进入不匹配的 collective,服务器因此挂死。

## 修复方案
- `scheduler.py`:删除 `_abort_on_waiting_timeout` / `_abort_on_running_timeout`(原先各 rank 本地删除请求/标记 `to_finish`),新增 `_poll_timeout_aborts`,仅扫描并产出 `AbortReq` 列表(waiting 与 running/last batch,PP>1 时扫 running_mbs/mbs,去重),不做任何本地删除;
- `request_receiver.py`:`recv_requests` 仅在 `pp_rank==0 && attn_tp_rank==0 && attn_cp_rank==0`(拥有 waiting queue 的 rank)调用 `poll_timeout_aborts`,产出的 AbortReq 随 `_broadcast_reqs_across_ranks` 既有广播走 work channel(注释说明 control channel 从 global rank 0 fan out,会覆盖除第一个 DP 组外的其他 DP 组的 aborts),每个 rank 在同一 scheduler iteration 经确定性 `abort_request()` 路径移除相同请求(waiting queue:pop;running batch:`to_finish`),不引入新 collective;
- `abort_request()` 现在传播发起方的 `finished_reason` / `abort_message`,客户端仍收到 503 "Request waiting/running timeout reached."(此前 waiting-queue echo 路径会丢掉原因)。

关键摘录:
```diff
+    def _poll_timeout_aborts(self) -> List[AbortReq]:
+        """Emit aborts only; every rank must drop the same requests in the
+        same iteration, or the extend-vs-decode decision splits and the
+        collectives hang."""
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围(与引入 per-rank 超时扫描的版本相关)。
- 回归面:超时语义与 env 默认值不变,仅 enforcement 机制移动。abort 现统一走 `abort_request()`,顺带获得其更完整的清理(disaggregation KV / metadata / beam retire);PP > 1 时注入的 aborts 与 tokenizer 发起的 AbortReq 同路径,但作者注明仅在 pp_size=1 上测试过;单测改为驱动 `_poll_timeout_aborts`,并新增 running 与 last batch 中同一请求的去重用例。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
