# PR #15439: [HiCacheStorage & PD] fix prefill bootstrap request host memory leaks

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15439](https://github.com/sgl-project/sglang/pull/15439) |
| 作者 | MOHENOO |
| 合并日期 | 2026-01-07 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 13 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论
PD 分离 prefill 侧 bootstrap 失败请求的 PrefetchOperation 所持 host_indices 无法释放,host 内存随时间越用越少;在 pop_bootstrapped 中对失败请求调用 `release_aborted_request` 释放预取事件,堵住泄漏。

## 问题现象
正文披露:prefill bootstrap 失败请求所持有的 PrefetchOperation 中的 host_indices 无法被释放,导致可分配的 host_indices 越来越小(allocatable host_indices become smaller and smaller),即 host 内存持续泄漏。触发场景是启用 hicache storage 的 PD(disaggregation)预填充端,请求在 bootstrap 阶段失败被 abort 时;正文附有两张监控截图佐证内存趋势。影响面为该配置下长期运行的 host 内存水位。

## 根因
正文披露:abort 的请求在 bootstrap 失败路径(`pop_bootstrapped` 收集 failed_reqs)中只做了计数,没有释放与该请求关联的预取操作;这些 PrefetchOperation 持有的 host_indices 永远不归还,host 内存池逐渐耗尽。因此必须释放 abort 请求的 prefetch operation 以防止 host 内存泄漏。

## 修复方案
改 `python/sglang/srt/disaggregation/prefill.py` 的 `pop_bootstrapped`:在失败请求(failed_reqs)分支中,若 `self.scheduler.enable_hicache_storage` 开启,则调用 `self.scheduler.tree_cache.release_aborted_request(req.rid)` 释放与该请求关联的 prefetch 事件。关键 diff 摘录:

```diff
                 failed_reqs.append(req)
                 if self.scheduler.enable_metrics:
                     self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
+                if self.scheduler.enable_hicache_storage:
+                    # to release prefetch events associated with the request
+                    self.scheduler.tree_cache.release_aborted_request(req.rid)
                 continue
```

## 影响与回归风险
- 回归面:仅 PD 分离预填充端的 bootstrap 失败处理路径,且被 `enable_hicache_storage` 开关守卫——未开启 hicache storage 的部署行为完全不变;开启后失败/abort 请求会额外走一次 `release_aborted_request`。回归风险主要是释放时机:若同一 rid 之后仍被预取流程引用,可能出现重复释放(正文未提及此类验证)。

## 关联
正文未列出(仅附两张 GitHub 用户附件截图,无 issue/PR 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
