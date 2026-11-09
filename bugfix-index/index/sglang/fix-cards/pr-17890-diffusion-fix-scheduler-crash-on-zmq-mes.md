# PR #17890: [diffusion]: fix scheduler crash on ZMQ messages with unexpected frame counts

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17890](https://github.com/sgl-project/sglang/pull/17890) |
| 作者 | Ratish1 |
| 合并日期 | 2026-02-17 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
多模态生成 scheduler 收到帧数不符的 ZMQ 消息即抛 ValueError 崩溃;改为弹性取帧、容忍坏包,保证事件循环不被无效流量打挂。

## 问题现象
多模态生成(multimodal generation)scheduler 的请求接收循环在收到"帧数不符合预期"的 ZMQ 消息时直接崩溃,抛出 ValueError。触发场景包括环境中存在网络扫描器或健康探针发来的无效流量——正文特别强调修复后 scheduler 能在这种环境下保持存活并响应。Fixes #17874。

## 根因
正文披露:原接收逻辑假定每条消息固定结构,`identity, _, payload = self.receiver.recv_multipart(zmq.NOBLOCK)` 按 3 帧解包,一旦帧数不符合预期(如探针/扫描器发的畸形消息)就抛 ValueError 使调度循环终止;此外 `pickle.loads(payload)` 对非 pickle 载荷也没有防护。

## 修复方案
改 `python/sglang/multimodal_gen/runtime/managers/scheduler.py` 的 `recv_reqs`:

```diff
-                    identity, _, payload = self.receiver.recv_multipart(zmq.NOBLOCK)
-                    recv_reqs = pickle.loads(payload)
-                except zmq.Again:
+                    # Accept valid REQ envelopes only, ignore malformed/probe frames.
+                    parts = self.receiver.recv_multipart(zmq.NOBLOCK)
+                    identity, payload = parts[0], parts[-1]
+
+                    # Ignore malformed probes or non-pickle data
+                    recv_reqs = pickle.loads(payload) if len(parts) > 2 else []
+                except (zmq.Again, pickle.UnpicklingError, IndexError, EOFError):
                     recv_reqs = []
```

即:弹性取帧(首帧作 identity、末帧作 payload);帧数 ≤2 视为畸形/探针消息直接忽略;异常防护扩展到 `pickle.UnpicklingError`、`IndexError`、`EOFError`,坏包不再终止事件循环。正文概括为三点:弹性帧提取以支持多种 socket 类型;对畸形/非 pickle 载荷加异常防护;保证扫描器/健康探针环境下 scheduler 存活。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 multimodal_gen scheduler 的 `recv_reqs`。行为变化:2 帧(≤2)消息从"崩溃"变为"忽略";>3 帧消息现在也会被解析(取首/末帧);畸形载荷静默丢弃,不再有任何告警入口——若真实请求因上游协议错误被发成坏包,将无提示地丢失。

## 关联
- Fixes #17874

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
