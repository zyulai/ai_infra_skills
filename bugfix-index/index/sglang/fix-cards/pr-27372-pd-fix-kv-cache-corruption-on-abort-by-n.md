# PR #27372: [PD] Fix KV cache corruption on abort by notifying ongoing prefill

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27372](https://github.com/sgl-project/sglang/pull/27372) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-06-05 |
| labels | (none) |
| 热度 | 👍 1 · 💬 4 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
PD 下长 ITL 请求 abort 后 prefill 侧不知情导致 KV cache 损坏;decode 侧尽力发送 ABORT 通知并跳过失效 chunk。

## 问题现象
PD disaggregation 场景下,长 ITL(inter-token latency)请求被 abort 时会触发竞态,造成 KV cache 损坏。正文称本修复「should address almost all race conditions caused by long ITL aborts」。正文未给出具体报错日志或复现命令,并说明有一类极端场景本 PR 覆盖不了:进行中的 prefill 已进入耗时极长的 transfer sync(最大值受 mooncake timeout 30s 限制),但该情况发生概率极低、极难复现。

## 根因
正文披露:该问题是 #24580 的轻量替代方案,针对长 ITL abort 引起的竞态——decode 侧请求 abort 后,prefill 侧并不知情,正在进行中的 prefill 继续为其工作,导致 KV cache 损坏(标题即「by notifying ongoing prefill」);修复思路是在 abort 时通知 prefill 侧。正文未进一步披露损坏的具体内存机制。

## 修复方案
- `python/sglang/srt/disaggregation/common/conn.py`:连接对象新增 `abort_notified` 标志;`abort()` 与 `_check_waiting_timeout()` 中在置 `KVPoll.Failed` 后调用新增的 `_send_abort_notification()`——对每个 bootstrap_info 尽力(best-effort)连接 bootstrap 服务器发送 `ABORT`(带 room、decode ip/port),失败仅记 debug 日志:
```python
sock.send_multipart(
    [b"ABORT",
     str(self.bootstrap_room).encode("ascii"),
     self.kv_mgr.local_ip.encode("ascii"),
     str(self.kv_mgr.rank_port).encode("ascii")]
)
```
- `python/sglang/srt/disaggregation/mooncake/conn.py`:prefill 侧 bootstrap 线程处理 `ABORT` 消息,room 未 Success 则标记 `KVPoll.Failed` 并回 `ABORT_ACK`;`transfer_worker` 跳过已失败/被 abort room 的 KV chunk;decode 线程处理 `ABORT_ACK`(留 TODO:如需要可据此实现 deferred release)。
- 正文未提供准确性/速度测试结果。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:PD disaggregation 的 common 与 mooncake 连接层。abort/等待超时路径新增一次尽力通知(失败静默);mooncake 传输循环对失败 room 跳过 chunk,正常传输路径多了两个状态检查。已完成的请求(Success)不受 abort 影响。遗留窗口:prefill 已进入超长 transfer sync 的场景仍可能命中正文所述的未覆盖竞态。

## 关联
- [#24580](https://github.com/sgl-project/sglang/pull/24580)(本 PR 为其轻量替代)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
