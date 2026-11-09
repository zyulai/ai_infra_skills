# PR #31857: [Bugfix] fix encoder cache leak of waiting requests in scheduler to solve stuck in CPU scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31857](https://github.com/vllm-project/vllm/pull/31857) |
| 作者 | frelam |
| 合并日期 | 2026-01-10 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论

多模态下 waiting 请求泄漏 encoder cache 塞满缓存,scheduler 死循环卡死 CPU;调度失败时手动释放修复。

## 问题现象

多模态推理时偶发调度卡死:所有请求都进入 `scheduler.py` 中 `num_new_tokens == 0` 的 `continue` 分支并死循环,卡在 CPU 调度上。进一步调试发现 encoder cache 不足:`encoder_cache_manager` 中 `num_embeds`(如 832)始终大于 `num_freeable_slots`(如 320),且 `freeable` 为空;更异常的是,大量**waiting 状态**的请求持有 encoder cache 条目(正文列出了 20 多个 chatcmpl-id 出现在 `cached` 字典中),而 waiting 请求本不应持有任何 encoder cache。于是 running 请求无法调度、waiting 请求也无法推进,形成调度死锁。

## 根因

正文未披露根本根因——作者明确表示这是用补偿性手段绕过,并在帖子里公开询问 root cause 可能是什么。正文给出的观测事实是:waiting 请求异常地持有 encoder cache(泄漏),导致 cache 被填满、无 freeable 槽位,调度死锁。从修复 diff 描述的机制看(由 diff 推断):在 `schedule()` 的 WAITING 路径中,当 `allocate_slots` 返回 `None`(请求无法调度)时,之前不会把该请求从 encoder cache manager 中释放,带 encoder 输入的请求于是持续占用缓存。

## 修复方案

改 `vllm/v1/core/sched/scheduler.py` 的 `schedule()`:当 waiting 请求分配槽位失败(`new_blocks is None`)且请求带 encoder 输入时,先把该请求从 encoder cache manager 中释放再 `break`:

```diff
                 if new_blocks is None:
                     # The request cannot be scheduled.
+
+                    # NOTE: we need to untouch the request from the encode cache
+                    # manager
+                    if request.has_encoder_inputs:
+                        self.encoder_cache_manager.free(request)
                     break
```

作者在正文中说明这只是补偿措施(compensatory measure),不是根治。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:v1 scheduler 的 WAITING 调度失败路径,仅影响带 encoder 输入(多模态/encoder-decoder)的请求;`free(request)` 只释放该请求本不应继续持有的缓存条目。若真正根因(谁把 cache 记到 waiting 请求头上)仍在,泄漏可能再次累积,只是现在每次调度失败都会清掉。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
