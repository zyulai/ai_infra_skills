# RFC #39760: [RFC]: Async Failure Notification for Fault Tolerant EP Kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39760](https://github.com/vllm-project/vllm/issues/39760) |
| 作者 | tzulingk |
| 创建日期 | 2026-04-14 |
| 状态 | ⚪ closed(关闭于 2026-08-14) |
| labels | stale |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |

## 一句话结论
为容错 EP kernel 的 GPU 侧失败检测设计异步通知机制,让同步代价只在失败真正发生时支付,保护 async scheduling 的吞吐收益。

## 提案动机
容错 EP kernel(如 NIXL-EP)在 GPU 上检测 rank 失败——超时经原子写更新 per-rank 整数掩码,无需 CPU 参与;CPU 只有显式把掩码读回才知道失败,这强制一次 GPU→CPU 同步。问题不在检测速度,而在同步的时机与频率:async_scheduling=True 时 CPU 把 pass N+1 的调度与 GPU 执行 pass N 流水化,阻塞式同步会摧毁该重叠——即使健康运行(无 rank 失败)也受害;"每次 forward 后检查"则每个 pass 都付代价。NIXL 团队指出:抛正式异常需把失败掩码读回 CPU,该同步点可能引入正想用 async run 避免的延迟。

## 提出的方案
- Design 1 —— Pinned Dirty Flag:GPU kernel 失败时向 pinned CPU 内存写一个整数 dirty_flag(st_release_sys_global);在 All2AllManagerBase.dispatch() 内检查(Option A,严格优于 pass 边界检查:检测延迟 ≤ 一个 MoE 层、浪费计算最少、异常栈自然);健康路径经 ctypes 直读 CPU RAM,开销可忽略;触发时一次 stream sync 排空 GPU、读全掩码定位失败 rank、重置 flag 并抛新异常 EPRankFailureError(携带 failed_ranks),由 worker.execute_model 捕获并发 ZMQ 失败通知;无需 C++,kernel 仅加一条 store
- Design 2 —— cudaLaunchHostFunc + ZMQ:把 host 回调排入 CUDA stream,CUDA 管理的后台线程在 combine 后查掩码并经 ZMQ 通知 engine core(其 asyncio 循环下一迭代处理 abort 与 _scale_down_elastic_ep);不抛异常、热路径零开销、无读回同步;但需要全量 pinned mask_cpu 数组与 C++ 扩展
- 结论:Design 1 务实先行,Design 2 是零热路径开销的终态
- 恢复路径复用 PR #38862 的三步(client 发 FinishReason.ABORT 中止在飞请求、DP 路由收缩 _scale_down_elastic_ep、expert 重分配/reload),两设计只是更早的触发器:GPU 掩码检测在一个 forward pass 内,而 #38862 的进程监控要等 OS 确认进程崩溃(可能数秒后)

## 优化点与收益
正文未给出量化数字;定性收益:健康路径开销"可忽略"(Design 1)/"零"(Design 2),掩码读回的同步只在失败时支付一次(Design 1)或完全消除(Design 2),检测延迟 ≤ 一个 MoE 层(Design 1)或 sub-pass(Design 2)。

## 现状与落地
已关闭(2026-08-14),labels 为 stale;正文未说明是否实现,落地情况需查关联 PR。文末附 5 项设计决策:dirty_flag_ptr 属各 All2AllManagerBase 子类内部细节、Design 2 的 C++ binding 放内核 buffer 库、新异常类型 EPRankFailureError(RuntimeError)、flag 由 CPU 在读全掩码后重置、多失败合并为一次检测事件。

## 关联
- PR #38862: https://github.com/vllm-project/vllm/pull/38862
- SchedulerConfig.async_scheduling(vllm/config/scheduler.py)、elastic_state.py 的 _BarrierTimeoutError 先例

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
