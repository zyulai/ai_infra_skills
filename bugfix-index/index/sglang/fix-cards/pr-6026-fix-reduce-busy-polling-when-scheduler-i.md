# PR #6026: [Fix] Reduce busy polling when scheduler is idle

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6026](https://github.com/sgl-project/sglang/pull/6026) |
| 作者 | p12tic |
| 合并日期 | 2025-06-12 |
| labels | ready-to-merge |
| 热度 | 👍 14 · 💬 9 |
| 分类 | Overlap / Scheduler |
| bug 类型 | general |

## 一句话结论
scheduler 空闲时各 GPU 各有一条线程 100% 忙等,新增 `--sleep-on-idle` 用 zmq.Poller 阻塞等待消息以降功耗。

## 问题现象
在存在长时间空闲期的部署中,sglang 空转时 scheduler 事件循环持续忙轮询,每块 GPU 都会将一条线程钉在 100% CPU 占用上,造成无谓功耗,并挤占 CPU 热余量(多 GPU 场景尤其明显)。主要受影响人群是向极少数用户提供 LLM 服务的住宅或小型商业用户。正文未描述报错日志等具体异常现象;本 PR 修复 issue #1730。

## 根因
PR 正文披露的动机是:scheduler 空闲循环不做任何阻塞等待,持续 busy polling,导致空闲时 CPU 仍被占满。正文没有更进一步的代码级根因分析;从 diff 看,各事件循环(normal/overlap/pp 及 disagg prefill/decode 变体)在 `batch is None`(空闲)分支里只做自检后立即进入下一轮循环,从未阻塞等待新消息(由 diff 推断)。

## 修复方案
新增可选参数 `--sleep-on-idle`(默认 False),在 scheduler 中引入 `IdleSleeper`:把所有需要立即处理消息的 socket 注册到 `zmq.Poller`,空闲时以 1000ms 超时阻塞 poll,替代忙轮询。改动文件:
- `python/sglang/srt/server_args.py`:新增 `sleep_on_idle: bool = False` 与 CLI 参数;
- `python/sglang/srt/managers/scheduler.py`:新增 `IdleSleeper` 类与 `maybe_sleep_on_idle()`;在 `recv_from_tokenizer`、`recv_from_rpc` socket 上注册 poller;在 `event_loop_normal`、`event_loop_overlap`、`event_loop_pp` 的空闲分支调用;
- `python/sglang/srt/disaggregation/prefill.py` / `decode.py`:四个 disagg 事件循环空闲分支同样调用;
- `docs/backend/server_arguments.md`:补文档。

关键 diff 摘录:

```python
class IdleSleeper:
    def __init__(self, sockets):
        self.poller = zmq.Poller()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

    def maybe_sleep(self):
        self.poller.poll(1000)
```

```diff
+                self.maybe_sleep_on_idle()
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:该参数默认关闭,不影响既有部署;开启后仅在服务器空闲分支额外阻塞 poll(最多 1000ms)。正文认为该代码只在空闲时运行、对性能副作用风险小;理论影响是空闲时新请求的响应需等 poll 返回(由 diff 推断,最多一个 poll 周期的延迟)。

## 关联
- [issue #1730](https://github.com/sgl-project/sglang/issues/1730)(正文 "FIXES #1730")
- [PR #1731](https://github.com/sgl-project/sglang/pull/1731)(本 PR 为其更简替代方案)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
