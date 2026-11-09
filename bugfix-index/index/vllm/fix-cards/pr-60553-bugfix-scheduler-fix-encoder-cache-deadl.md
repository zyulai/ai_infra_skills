# PR #60553: [Bugfix][Scheduler] Fix encoder cache deadlock with drafter look-ahead

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60553](https://github.com/vllm-project/vllm/pull/60553) |
| 作者 | gty111 |
| 合并日期 | 2026-10-19 |
| labels | bug, ready, scheduler |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
调度器两处按 drafter look-ahead L 反向偏移导致 encoder cache 死锁(请求 0 调度挂死),两处同时去掉 L 偏移修复。

## 问题现象
EAGLE/MTP 投机解码 + 多模态:同一请求两个多模态项一次只能装进 encoder cache 一个、且两者间隔 gap 小于 `2 * num_prefill_lookahead`(记 L)时,请求永久挂起——每步调度到 0 个 token,无任何报错或日志。正文 E2E 复现:1x GB200、MRV2、async scheduling、prefix caching 关闭,`Qwen/Qwen3.5-4B` MTP `num_speculative_tokens=1`(L=1),每个请求两张 3008x4032 图像(各 11844 token),encoder cache 16384 只能容纳一张;main 分支在 gap=0、gap=1 时 300 秒内不完成(hang),gap=2 正常;本 PR 修复后全部 OK 且 temperature 0 输出与无投机基线逐字一致,未见 `Encoder cache miss`。调度器级 sweep(L x gap)显示 main 在 L=1、gap<2 及 L=3、gap<8 时 hang。#40707 的 Mamba-alignment 部分已由 #51603 修复,本 PR 修其 encoder-cache 死锁部分。

## 根因
正文披露:两处代码被 drafter look-ahead `L` 以相反方向偏移:
1. `_free_encoder_inputs` 把多模态项 1 保留到 `num_computed_tokens >= end1 + L` 才释放(该偏移由 #46315 引入);
2. 项 2 无法分配时,`_try_schedule_encoder_inputs` 把 chunk 停在 `start2 - L`。
若 `start2 - L < end1 + L`,`num_computed_tokens` 永远达不到释放项 1 的条件,项 2 永远无法分配,形成死锁。

## 修复方案
`vllm/v1/core/sched/scheduler.py` 两处同时去掉 L 偏移:
- 释放条件改回与非投机路径一致:`start_pos + num_tokens <= num_computed_tokens - num_output_placeholders` 即释放(拒绝回滚仍由 `num_output_placeholders` 覆盖;V1/V2 的 drafter gather 只读 `[c+1, c+k+1)` 窗口,越过项后目标与 drafter 都不会再读它)。
- `_try_schedule_encoder_inputs` 的 `can_allocate` 失败分支停在 `start2`(不再减 L),drafter 的 read-ahead 落在未编码项上时,两个 runner 已有 `start_pos >= computed + scheduled` 检查回退到 token embeddings(#46315 引入);`disable_chunked_mm_input` 回滚仍保留 shift。
```python
-                if num_computed_tokens + shift_computed_tokens < start_pos:
+                if num_computed_tokens < start_pos:
                     ...
-                    num_new_tokens = start_pos - (
-                        num_computed_tokens + shift_computed_tokens
+                    num_new_tokens = min(
+                        num_new_tokens, start_pos - num_computed_tokens
                     )
```
正文强调两处缺一不可,单独改任一处只能把挂死条件从 `gap < 2L` 收窄到 `gap < L`。权衡:cache 满时 drafter 在 chunk 边界最多 L 个位置用 token embeddings 代替 image embeddings,可能使单步接受率略降;项能装进 cache 的请求不受影响。

## 影响与回归风险
- 回归面:V1 调度器的 encoder input 释放时机(`_free_encoder_inputs`)与多模态 chunk 调度截断(`_try_schedule_encoder_inputs`),影响开投机解码的多模态请求;输出不变(正文验证 temperature 0 与基线一致)。cache 紧张时 drafter 边界位置退化为 token embedding,单步接受率可能略降。

## 关联
- #40707(fixes,含[根因评论](https://github.com/vllm-project/vllm/issues/40707#issuecomment-5760731067))
- #51603(Mamba-alignment 部分,已修)
- #46315(引入 L 偏移与 token embedding 回退)
- #59030(中途中释放方案,与本 PR 互补但未解此死锁)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
