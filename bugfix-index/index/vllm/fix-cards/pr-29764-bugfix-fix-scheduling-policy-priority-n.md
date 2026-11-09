# PR #29764: [Bugfix] fix --scheduling-policy=priority & n>1 crashes engine

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29764](https://github.com/vllm-project/vllm/pull/29764) |
| 作者 | chaunceyjiang |
| 合并日期 | 2025-12-02 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论

优先级队列在请求同优先级同到达时间时崩溃,给 Request 实现 __lt__ 修复。

## 问题现象

以 `vllm serve /home/qwen3-8b --scheduling-policy=priority` 启动服务,再用 OpenAI 客户端发起 `n=2` 的 chat completion 请求,引擎崩溃。崩溃栈(正文照抄):

```
(EngineCore_DP0 pid=28688)   File ".../vllm/v1/core/sched/request_queue.py", line 152, in add_request
(EngineCore_DP0 pid=28688)     heapq.heappush(self._heap, (request.priority, request.arrival_time, request))
(EngineCore_DP0 pid=28688) TypeError: '<' not supported between instances of 'Request' and 'Request'
```

修复后同样的请求返回 200 OK,引擎正常出吞吐日志("Running: 2 reqs")。影响面:所有使用 `--scheduling-policy=priority` 且请求带 n>1 的部署。

## 根因

正文未以文字披露根因,但正文中的崩溃栈显示:TypeError 发生在 `PriorityRequestQueue.add_request` 的 `heapq.heappush(self._heap, (request.priority, request.arrival_time, request))` 上,比较失败发生在元组的第三个元素——两个 Request 对象之间。(由 diff 推断)n>1 时派生出的子请求 priority 与 arrival_time 完全相同,元组前两元素相等后 heapq 退化为比较 Request 对象本身,而 `Request` 类未定义 `__lt__`,堆排序直接抛 TypeError。

## 修复方案

两处改动:

1. `vllm/v1/request.py`:给 `Request` 新增 `__lt__`,按 priority → arrival_time → request_id → id(self) 逐级比较,保证任意两个请求可比较、排序全序确定。
2. `vllm/v1/core/sched/request_queue.py`:`PriorityRequestQueue` 不再把 `(priority, arrival_time, request)` 三元组塞进堆,改为直接压入/弹出 Request 对象,依赖新 `__lt__` 排序;`remove_request`/`remove_requests`/`__iter__` 等同步简化。

关键 diff 摘录(vllm/v1/request.py):

```python
def __lt__(self, other: "Request") -> bool:
    if self.priority != other.priority:
        return self.priority < other.priority
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    if self.request_id != other.request_id:
        return self.request_id < other.request_id
    return id(self) < id(other)
```

另在 `tests/v1/core/test_priority_scheduler_random.py` 的随机压测中新增 2 条初始请求,其 priority 与 arrival_time 范围均为单点(4,4)/(0,0),专门覆盖同优先级同到达时间的并列场景。

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅优先级调度路径(`--scheduling-policy=priority`)。堆内排序键从三元组变为 Request 自身比较,新增 request_id 与 id(self) 两个 tiebreaker,完全相同优先级/到达时间的请求之间出队顺序可能与旧实现不同(旧实现此时未定义/崩溃);非 priority 调度使用普通 FIFO 队列,不受影响。

## 关联

- [issue #29747](https://github.com/vllm-project/vllm/issues/29747)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
