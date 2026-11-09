# RFC #18826: [RFC]: Controlling the maximum length of the waiting queue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18826](https://github.com/vllm-project/vllm/issues/18826) |
| 作者 | chaunceyjiang |
| 创建日期 | 2025-05-28 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 18 · 💬 11 |
| 分类 | Other |

## 一句话结论
新增 --max-waiting-queue-length 参数,等待队列满时立即以 HTTP 503 拒绝新请求,防止内存无界增长。

## 提案动机
vLLM 目前没有按等待队列长度拒绝请求的机制,所有进来的请求都被加入队列;等待队列是 CPU 内存中的无界 deque(scheduler.py),每个元素是一个待处理请求。在服务过载或使用低性能 GPU 的场景下,若用户不主动取消,队列长度可能无限增长,造成内存消耗过大并最终 OOM。

## 提出的方案
- 新增参数 `--max-waiting-queue-length`:scheduler 尝试把请求加入等待队列时,先检查队列是否已达最大长度;满了就拒绝请求,返回 HTTP 503 (Service Unavailable)。
- 实现路径:队列满时 `add_request` 抛出 `SchedulerWaitingQueueFullError`(正文附伪代码);EngineCore 经 ZMQ 把该错误发给 API server,API server 检测到此错误后把 HTTP 状态码置为 503。
- 更高层配合:在 Kubernetes 等环境中,可用 Istio、Envoy 等工具基于 503 状态码做负载均衡。
- 补充考虑:可复用 PR #17546 新增的 `get_request_counts` 接口;DP 模式下多个 EngineCore 各自维护等待队列,或可在 DP Coordinator 层统一处理(作者自注仅为猜测)。

## 优化点与收益
正文未给出量化数字。定性收益:过载时以快速失败替代无界排队,避免内存耗尽与 OOM;503 信号可被服务网格用于负载均衡分流。

## 现状与落地
仍在讨论/待实现(open),CC 了 @DarkLight1337、@simon-mo、@njhill;正文提到已有多个 issue 请求此特性(如 #2901);具体落地需查后续关联 PR。

## 关联
- [scheduler.py 等待队列实现](https://github.com/vllm-project/vllm/blob/6825d9a998df3f22b5e19fb600d0a0c09950db28/vllm/v1/core/sched/scheduler.py#L90-L92)
- [PR #17546](https://github.com/vllm-project/vllm/pull/17546)(get_request_counts 接口)
- [#2901](https://github.com/vllm-project/vllm/issues/2901)(同类特性请求)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
