# PR #21514: [rl][sgl] fix tensor mismatch after pause

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21514](https://github.com/sgl-project/sglang/pull/21514) |
| 作者 | bixue2010 |
| 合并日期 | 2026-03-27 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

pause_generation 无条件 merge 已过滤为空的 last_batch,破坏 reqs/tensors 不变量致 shape mismatch,补 is_empty 守卫修复。

## 问题现象

RL 场景调用 `pause_generation` 后触发 tensor 形状不匹配,表现为 `grouped_foreach_copy` 中的错误:

```
RuntimeError: The size of tensor a (652) must match the size of tensor b (651)
```

触发条件:extend 的 last_batch 中所有请求都已 finished 时执行 pause_generation。多余的槽位随后破坏 KV-cache 分配,最终以上述 shape mismatch 形式暴露。

## 根因

正文披露完整根因链:

1. `pause_generation` 在对 last_batch 调用 `filter_batch` 后无条件调用 `merge_batch`,而 `get_next_batch_to_run` 中的同段逻辑带 `is_empty()` 守卫。
2. 当 extend last_batch 全部请求 finished 时,`filter_batch` 置 `last_batch.reqs = []` 后提前 return,不更新 tensors(seq_lens、req_pool_indices、output_ids 等仍保持 M=1 个元素)。
3. `merge_batch` 随后将这些陈旧的 M 元素张量 torch.cat 进 running_batch,得到 `seq_lens.shape[0] = N+1` 而 `len(reqs) = N`,破坏 `len(reqs) == seq_lens.shape[0]` 不变量。
4. 下一次 `update_running_batch` 中的 `filter_batch` 按 `range(len(reqs))=range(N)` 计算 keep_indices,发现 `len(keep_indices)==N==len(reqs)` 又提前 return,不会修正 tensor 尺寸,不一致持续存在。

## 修复方案

在 `python/sglang/srt/managers/scheduler.py` 的 `pause_generation` 中,把无条件的 `self.running_batch.merge_batch(self.last_batch)` 替换为与 `get_next_batch_to_run`(line 2149)相同的 `is_empty()` 守卫逻辑。

关键 diff 摘录:

```diff
             self.last_batch.filter_batch(
                 chunked_req_to_exclude=list(chunked_req_to_exclude)
             )
-            self.running_batch.merge_batch(self.last_batch)
+            if not self.last_batch.is_empty():
+                if self.running_batch.is_empty():
+                    self.running_batch = self.last_batch
+                else:
+                    self.running_batch.merge_batch(self.last_batch)
```

同时新增单测 `test/registered/rl/test_pause_generation_tensor_consistency.py`(212 行),用最小化的 ScheduleBatch 逻辑模拟复现不变量破坏与修复后的各分支(全 finished、部分存活、running_batch 为空)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 `pause_generation` 中 last_batch 向 running_batch 合并的路径;行为与 `get_next_batch_to_run` 已有逻辑对齐(last_batch 为空则跳过 merge,running_batch 为空则直接替换)。对正常非空 merge 无影响,风险集中在 pause/resume(RL rollout)流程。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
