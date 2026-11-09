# PR #1767: [Fix] Fix abort in data parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1767](https://github.com/sgl-project/sglang/pull/1767) |
| 作者 | merrymercy |
| 合并日期 | 2024-10-23 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Parallelism / Communication |
| bug 类型 | crash |

## 一句话结论
DP controller 广播控制消息(如 abort)误用不可访问的 worker.queue,报 AttributeError,改用 send_pyobj。

## 问题现象
数据并行(data parallelism)下执行 abort 等控制消息广播时,报错 `AttributeError: queue attribute is write-only`。正文只有这一句,未描述触发条件的更多细节;结合标题与 diff 位置,该错误发生在 DP controller 的 `event_loop` 向所有 worker 发送控制消息(abort 属于此类「other control messages」)的路径上,导致 abort 流程失败。影响面为启用数据并行的部署。

## 根因
正文披露的直接错误是 `AttributeError: queue attribute is write-only`,未展开更深层原因。

从 diff 看,出错代码位于 `data_parallel_controller.py` 的 `event_loop`:对除 dispatch 之外的控制消息,原来通过 `worker.queue.put(recv_req)` 逐个投递给所有 worker;而该 worker 对象上的 `queue` 属性在此场景下是「write-only」(读取/访问即抛 AttributeError)。修复改为 `worker.send_pyobj(recv_req)`,即走与常规请求分发一致的 pyobj 发送通道。可推断机制是:worker 引用并非暴露可读 queue 的对象,`queue.put` 访问失败,而 `send_pyobj` 是该 worker 支持的通信接口(由 diff 推断)。

## 修复方案
改动 `python/sglang/srt/managers/data_parallel_controller.py` 的 `event_loop`,共一行:

```python
else:
    # Send other control messages to all workers
    for worker in self.workers:
        worker.send_pyobj(recv_req)
```

即控制消息广播从 `worker.queue.put(recv_req)` 换成 `worker.send_pyobj(recv_req)`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 DP controller 的控制消息广播路径(abort 等非 dispatch 消息);常规请求分发路径与 worker 侧处理不变。若存在依赖 `worker.queue` 可写语义的调用方,需确认其使用方式,但 diff 范围内未见其他改动。

## 关联
正文未列出(未附 issue/PR 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
