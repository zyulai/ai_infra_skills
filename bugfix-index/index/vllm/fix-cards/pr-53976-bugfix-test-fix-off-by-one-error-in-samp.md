# PR #53976: [Bugfix][Test] Fix off-by-one error in sampled token rank causing flaky logprobs test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53976](https://github.com/vllm-project/vllm/pull/53976) |
| 作者 | mayuyuace |
| 合并日期 | 2026-08-31 |
| labels | bug, ready, verified |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

测试工具 `_create_random_top_token_test_vector` 用 0-based 索引当 rank,与 logprobs.py 的 1-based 约定差一,当采样 token 在 top-k 中重复出现时断言 flaky;加 1 修复。

## 问题现象

`tests/v1/engine/test_output_processor.py::test_logprobs_processor[None-5-RequestOutputKind.DELTA]` 出现非确定性(偶发)失败,Intel CI 上可复现(Buildkite build 9778)。断言 `assert ref_sampled_token_rank == smp_lp_rank`(test 第 293 行)偶发不成立。概率约为每 token 位置 `num_logprobs / vocab_size`(如 5/128000 ≈ 0.004%),torch 随机种子未固定,大多数时候能通过,但长期在 CI 中反复出现。

## 根因

正文披露:`tests/v1/engine/utils.py` 的 `_create_random_top_token_test_vector()` 构造含 `sampled_token_rank` 的测试夹具。用 `torch.randperm` 生成 `choice_tensor` 后把 `choice_tensor[0]` 强制设为 `sampled_token_id`;若 `randperm` 恰好把 `sampled_token_id` 也放进 `choice_tensor[1:]`,函数把 `(choice_tensor[1:] == sampled_token_id).nonzero(...)` 的 0-based 索引直接当作 rank 返回。而 `vllm/logprobs.py::append_logprobs_for_next_position` 用 `range(1, num_logprobs + 1)` 赋 1-based top-k rank;当 `sampled_token_id` 在 token_ids 中出现两次(索引 0 和 top-k 内某处)时,dict 条目被 1-based top-k rank 覆盖,而 `ref_sampled_token_rank` 保存 0-based 值,两者差 1 导致断言失败。

## 修复方案

修改 `tests/v1/engine/utils.py`(仅测试工具,无生产代码):两处 0-based 索引转 1-based。关键 diff:

```diff
-        sampled_token_rank = (
-            (choice_tensor[1:] == sampled_token_id).nonzero(as_tuple=True)[0].item()
-        )
+        # Use 1-based rank to match the top-k rank convention used by the
+        # consumer (logprobs.py assigns ranks starting from 1).
+        sampled_token_rank = (choice_tensor[1:] == sampled_token_id).nonzero(
+            as_tuple=True
+        )[0].item() + 1
```

`_create_random_top_token_test_matrix` 中 `prompt_token_ranks[rdx] = token_index.item()` 同样改为 `+ 1`。作者另提供独立复现脚本,确定性构造最坏情况验证修复前后行为(2 != 3 → 3 == 3)。

## 影响与回归风险

- 影响窗口:正文未说明;仅影响测试代码。
- 回归面:仅 `tests/v1/engine/utils.py` 的两个测试夹具函数,生产代码零改动;rank 语义与消费方(logprobs.py 的 1-based 约定)对齐,不影响任何运行时路径。

## 关联

- https://buildkite.com/vllm/intel-ci/builds/9778/canvas?sid=01a03c1d-1b9a-43d7-8c99-b01d13ceabdc&tab=output

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
