# PR #9927: fix(cache): move ongoing_prefetch pop after validation to prevent leak

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9927](https://github.com/sgl-project/sglang/pull/9927) |
| 作者 | xiaguan |
| 合并日期 | 2025-09-03 |
| labels | ready-to-merge |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论

HiRadixCache 在校验前就 pop `ongoing_prefetch` 条目导致 prefetch 泄漏、hicache 失灵;改为先查询、校验通过释放后再 `del`。

## 问题现象

正文称这是一个高优先级 bug:所有 prefetch 操作都会泄漏——prefetch 操作无法被释放,进而导致 hicache 失灵(内存未被释放),且由于 prefetch limit 持续被触发,后续 prefetch 无法进行。正文未给出具体复现命令或版本号;场景为 hicache/HiRadixCache 的 prefetch 路径。

## 根因

正文披露的影响是 prefetch 操作泄漏且无法释放;但未进一步披露代码层面的机制,仅标题指明修复方向是「把 ongoing_prefetch 的 pop 移到校验之后」。(由 diff 推断)原代码在 `check_prefetch_progress` 入口处即执行 `self.ongoing_prefetch.pop(req_id)`,若随后的校验未通过而提前返回,已被取出的条目连同其 host 内存与占用计数都不会被释放;`release_aborted_request` 也有同样的先 pop 后处理模式。

## 修复方案

改 `python/sglang/srt/mem_cache/hiradix_cache.py` 两处,均把「入口 pop」改为「入口按 key 取值 + 成功路径末尾 del」:

`check_prefetch_progress`:

```diff
-        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch.pop(
+        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
             req_id
-        )
+        ]
         ...
         last_host_node.release_host()
+        del self.ongoing_prefetch[req_id]
         self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
```

`release_aborted_request` 同样:入口 `pop(rid)` 改为 `self.ongoing_prefetch[rid]`,在 `last_host_node.release_host()` 之后新增 `del self.ongoing_prefetch[rid]`,再做 `append_host_mem_release` 与 `prefetch_tokens_occupied` 扣减。这样校验未通过时条目保留在 `ongoing_prefetch` 中,释放动作完成后才移除条目。

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围;受影响路径为 `hiradix_cache.py` 的 prefetch 进度检查与中止请求释放逻辑。
- 回归面:改动仅涉及 HiRadixCache 的 `ongoing_prefetch` 生命周期管理。成功路径行为不变(释放后移除条目);中止请求若 `host_indices is None` 提前返回时条目不再被移除,依赖该 dict 收敛的代码需注意;其他 cache 后端不受影响。

## 关联

正文未列出 issue/PR 链接(仅 @ 提及 xiezhq-hermann)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
