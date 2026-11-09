# PR #23006: [Pipeline Parallelism][Bug] Fix scheduler hang in pipeline parallelism setup

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23006](https://github.com/sgl-project/sglang/pull/23006) |
| 作者 | jonahbernard |
| 合并日期 | 2026-04-17 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
PP 服务器上分块 prefill 占满最后一个 req 槽后,`_get_new_batch_prefill_raw` 因 chunked_req 判断被 #14883 反转而在每轮事件循环都返回 None,服务器 hang;把条件改回 `self.chunked_req is None`。

## 问题现象
正文披露:使用 pipeline parallelism 的 SGLang server 对 prefill 请求 hang,原因是 `_get_new_batch_prefill_raw` 在主调度器事件循环的每一轮都返回 `None`。复现命令(以 Qwen/Qwen3-30B-A3B、pp_size=2 为例):`python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --moe-runner-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --chunked-prefill-size 64 --max-running-requests 1 --port 30004 --pp-size 2`,发送单个大 prompt 即可触发(大 prompt 被切成 chunks)。影响面:PP 部署且空闲槽位紧张时,新请求与当前 chunked 请求都无法推进,「server will hang for lots of users」。

## 根因
正文披露了完整推导:PP=2、max_running_requests=1 时 `pp.max_micro_batch_size=1`;req_to_token_pool 起始仅 1 个空闲槽。大 prompt 分块后 `Req` 被记为 `self.chunked_req` 并分配走该槽,`len(free_slots)=0`,于是 `get_num_allocatable_reqs = min((1 - running_bs), 0) = 0`,使下面这段判断每轮都为真并 `return None`:

```python
if (
    self.get_num_allocatable_reqs(running_bs) <= 0
    and self.chunked_req is not None
    and not self.enable_priority_preemption
):
    self.running_batch.batch_is_full = True
    return None
```

该判断的语义本意是「无空闲槽且不存在 chunked 请求时才把批次标记为满」——chunked 请求应始终允许加入新批次,否则内存泄漏(代码注释原文)。历史:PP 初次引入(PR #5724)时条件为 `not self.chunked_req`;PR #14883 把它悄然反转为 `self.chunked_req is not None`(两者逻辑相反),从此有 chunked 请求且槽满时必然 hang。

## 修复方案
改 `python/sglang/srt/managers/scheduler.py` 的 `_get_new_batch_prefill_raw`,把条件 `self.chunked_req is not None` 改回 `self.chunked_req is None`。关键 diff 摘录:

```diff
         if (
             self.get_num_allocatable_reqs(running_bs) <= 0
-            and self.chunked_req is not None
+            and self.chunked_req is None
             and not self.enable_priority_preemption
         ):
```

## 影响与回归风险
- 影响窗口:自引入该反转的 PR #14883 合并之后、本 PR 之前的版本。
- 回归面:仅 `scheduler.py` 中这一处布尔条件,语义恢复为「槽满且无 chunked 请求才置 batch_is_full 并跳过」;chunked prefill 在槽满时恢复可见进度。风险在于恢复后 `get_num_allocatable_reqs<=0 且存在 chunked_req` 的场景会继续往新批次塞 chunked 请求(原 #5724 设计,注释称这是避免内存泄漏所需)。

## 关联
- https://github.com/sgl-project/sglang/pull/5724 (PP 初次引入,原判断出处)
- https://github.com/sgl-project/sglang/pull/14883 (引入反转、造成 bug 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
