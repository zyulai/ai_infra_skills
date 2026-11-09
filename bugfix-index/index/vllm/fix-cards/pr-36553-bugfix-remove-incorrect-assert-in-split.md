# PR #36553: [BugFix] Remove incorrect assert in split_decodes_and_prefills

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36553](https://github.com/vllm-project/vllm/pull/36553) |
| 作者 | WoosukKwon |
| 合并日期 | 2026-03-10 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

`split_decodes_and_prefills` 的 padding 校验 assert 未考虑 `query_lens == 0` 的合法情况,误报崩溃,直接移除。

## 问题现象

当 batch 中存在 `query_lens == 0` 的请求(纯 padding,例如某些请求没有新 query token)时,`vllm/v1/attention/backends/utils.py` 的 `split_decodes_and_prefills` 会触发断言 `"tokens not padded correctly"` 而中断。正文未给出具体复现命令或日志,但指明触发条件是该 assert 未考虑 `query_lens == 0` 的场景。

## 根因

PR 正文披露:该 assert 语句没有考虑 `query_lens == 0` 的情况。断言检查 `num_reqs * query_lens[0] == num_tokens`(假设所有请求 query 长度都等于第一个请求),但当部分请求 query 长度为 0(padding)时,该等式天然不成立——而按代码注释,这种情况是合法的:这些请求可以安全地当作 decode 处理(以保证 `num_decodes` 与 captured size 匹配),因此 assert 本身是错误的检查。

## 修复方案

改 `vllm/v1/attention/backends/utils.py` 的 `split_decodes_and_prefills`:删除「全为相同 query 长度或 0」分支中的错误断言,保留原有返回逻辑(全部按 decode 处理)。

```diff
         if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
-            assert num_reqs * query_lens[0] == num_tokens, "tokens not padded correctly"
             return num_reqs, 0, num_tokens, 0  # all decodes
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 `split_decodes_and_prefills` 中「全 decode」分支的一个检查;函数输出不变,只是不再对该分支做 padding 一致性断言。任何依赖此 assert 提前发现真正 padding 错误的排查场景会失去这一检查点,属预期取舍。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
