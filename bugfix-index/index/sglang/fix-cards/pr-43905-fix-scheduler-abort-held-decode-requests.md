# PR #43905: [Fix] scheduler: Abort held decode requests, and read live requests from the lists

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43905](https://github.com/sgl-project/sglang/pull/43905) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-11-03 |
| labels | Multi-modal, run-ci, apple-silicon |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
abort 触达不到 held decode 请求、多处自遍历队列漏请求;让 abort 覆盖 held 请求并统一改读 request_lists()。

## 问题现象
在已有 `Scheduler.request_lists()` 统一存活请求列表的情况下,若干代码仍各自遍历自己的队列/批次子集,要么漏掉请求、要么重复罗列 holders。具体后果:PD decode 请求在 generation 暂停期间为 rebootstrap 而 held 的(`DecodePreallocQueue.held_rebootstrap_reqs`)无法被 `/abort_request` 触达(被删的旧注释直言 "requests held here are not reachable by /abort_request",并称 RL 场景下不应出现);`_has_active_sampling_mask_request` 只看自己的批次/队列集合,看不到 PD prefill bootstrap 与 transfer 队列、已启动的 prefill、dLLM 队列、正在释放 KV 的 PD decode 请求、HiSparse staging 以及 parked uOCR windows 中的请求。

## 根因
正文披露:这些调用点没有从 `request_lists()` 读取统一列表,而是各自维护子集遍历,导致 abort 无法到达被跳过的列表(held rebootstrap 请求),采样掩码检查漏掉部分持有请求的列表。

## 修复方案
- `disaggregation/decode.py`:`DecodePreallocQueue` 新增 `take_aborted(recv_req)`,按 `abort_all` 或 rid 前缀匹配,从 `retracted_queue` 与 `held_rebootstrap_reqs` 中取出被 abort 的请求并从两个列表移除;
- `scheduler.py::abort_request`:PD decode 分支改用 `take_aborted`——retracted 请求(持有 KV backup)照旧 discard backup 并发 abort 输出;held 请求(不持有 KV、无 step 使用)立即 `finish_outside_step(FINISH_ABORT())`、释放,并发送带 `finished_reason` 的 abort 输出;
- `internal_state.py::_has_active_sampling_mask_request`:删除手工拼装的批次/队列清单,改为遍历 `request_lists()` 的全部列表;
- `scheduler.py::is_fully_idle`:改从 `request_lists()` 的请求部分计算(default / health check / `ignore_waiting` 三种变体),default 变体排除 running、prefill_launched、dllm、decode_held、uocr_parked、deferred 等已在别处计数或有意不计的列表,health check 仍计 chunked、waiting 及未暂停时的 retracted decode;正文说明其结果不变。

新增单测 `test_pd_decode_abort_finishes_a_held_rebootstrap`:retract 暂停期间 abort 只结束匹配请求,continue_generation 后其余 held 请求重新入队。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:行为改变有两处——held 请求现在可被 abort 并立即释放;采样掩码检查覆盖更多列表(此前遗漏列表中的 mask 请求现在会被看到,可能影响输出/回退决策)。`is_fully_idle` 按正文说明语义等价,但其重写波及 flush_cache/health check 等依赖 idle 判断的路径;PD decode retracted/held 队列的 abort 清理路径整体重构,相关单测夹具同步更新。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
