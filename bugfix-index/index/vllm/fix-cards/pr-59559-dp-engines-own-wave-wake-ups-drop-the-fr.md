# PR #59559: [DP] Engines own wave wake-ups; drop the front-end FIRST_REQ

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59559](https://github.com/vllm-project/vllm/pull/59559) |
| 作者 | aoshen02 |
| 合并日期 | 2026-10-16 |
| labels | bug, ready, rust |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
前端 FIRST_REQ「以为唤醒成功」的假设在暂停 peer 上失效致 DP 永久阻塞;改为仅由引擎发起 wave。

## 问题现象
vLLM 数据并行(DP)场景下的 hang。触发条件(正文给出的最小复现):两个 API server(A、B)与两个 rank(X、Y)全部暂停时——1) B 向 X 发 ADD、向前端协调器发 `FIRST_REQ`;2) 协调器向 Y 发 `START_DP_WAVE`,Y 仍处于暂停态,直接丢弃;3) A 恢复两个 rank,两 rank 均无任务,保持 idle;4) B 的 ADD 到达 X,wave 是当前的,X 因此不 announce、独自 step,在 DP collective 中永久阻塞。根子是 B→X、coordinator→Y、A→X/Y 三条 socket 之间没有任何顺序保证。正文测试表中,main 分支上多处 DP 用例 hang(如 ADD 跨 pause/resume 持有、暂停期间发请求后 resume 的首请求 45 秒超时),本 PR 后全部通过。

## 根因
正文披露:旧设计中前端在「自认为引擎空闲」时发送 `FIRST_REQ`,协调器随后广播 `START_DP_WAVE`,而收到请求的 rank 想当然认为 peer 已被唤醒。该假设在 peer 仍暂停时失败——暂停的 rank 会丢弃 `START_DP_WAVE`;且前端乐观置位的 running 标志可能过期(main 上每个 wake 平均广播两次 `START_DP_WAVE` 即其佐证)。正文本着「只有即将 step 的 rank 才知道自己要 step」重构:空闲且未暂停的 rank 收到请求时总是 announce `start_wave`,协调器去重后广播给其他 rank,前端唤醒随之失去正确性职责。正文同时指出这是从协调器侧移除同一假设的 vllm-project/vllm#51481 的延续。

## 修复方案
净改动 12 个文件,+226/−531。要点:
- 引擎(`vllm/v1/engine/core.py` `DPEngineCoreProc.add_request`):空闲、未暂停的 rank 收到请求即 announce,不再仅限 stale wave:

```diff
-        if self.has_coordinator and request_wave != self.current_wave:
+        if self.has_coordinator:
             if request_wave > self.current_wave:
                 self.current_wave = request_wave
-            elif (
+            if (
                 not self.engines_running
                 and self.scheduler.pause_state == PauseState.UNPAUSED
             ):
+                # Engines alone start waves: have the coordinator wake the others.
                 self.engines_running = True
```

- Python 前端(`vllm/v1/engine/core_client.py` `DPAsyncMPClient`):删除 `FIRST_REQ` 发送与乐观 `engines_running` 置位;PAIR socket 仅承载 elastic EP scale 通知并更名 `scale_*`。
- Python 协调器(`vllm/v1/engine/coordinator.py`):移除前端 wake 处理分支,收到前端消息改为记录 error 日志。
- Rust 前端:删除 `CoordinatorCommand::FirstRequest`、`notify_first_request`、命令 channel 与外部唤醒消息;in-process 协调器只转发引擎的 `StartWave`/`WaveComplete`。
- 保持不变:暂停 rank 仍丢弃 `START_DP_WAVE`(`ignore_start_dp_wave`,新增单测固化)、rank 0 在 collective 后的 `start_wave`、请求的 `current_wave` 戳与协调器去重。

正文验证覆盖单测、DP(mp/Ray)、三种 LB 模式、多 API server、PP+DP、elastic EP、三种前端各 10 次冷启动 + 3 轮 16 请求突发 + 4 轮 pause/resume,全部通过;空闲 TTFT 与 main 持平(差异 < 1–2 ms 轮间波动);代价是 remote node 上每次 idle→running 转换多约两次跨节点单程唤醒开销(仅 idle 后首个请求支付)。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动横跨 Python 引擎/协调器/前端与 Rust 引擎客户端/协调器/前端,是 DP wave 控制面的行为变更。兼容性(正文):新前端需要带本改动的引擎(两者同发布,正常部署总是满足);旧 Rust 前端配新引擎仍可用,协调器会记录并忽略其 wake 消息。性能:DP=16 在 4 req/s Poisson 负载下 p50 慢 1.5–2 ms(仅 remote node 首 wakes),DP=8 无差异;每次 wake 从两次广播降为一次。

## 关联
- vllm-project/vllm#51481(协调器侧移除同一假设)
- vllm-project/vllm#51488(本 PR 由其 review 引出,#51488 现基于本 PR)
- vllm-project/vllm#54762、vllm-project/vllm#55229(触及 DP wave 状态但不动唤醒路径,正文用于说明非重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
**注:pack 中的 diff 在 40000 字符处被截断(正文称共 12 个文件,截断前可见 9 个文件),截断部分未在卡中引用。**
