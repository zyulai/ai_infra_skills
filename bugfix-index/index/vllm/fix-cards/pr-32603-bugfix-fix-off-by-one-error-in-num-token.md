# PR #32603: [Bugfix] Fix Off-by-one error in _num_tokens_to_min_blocks calculation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32603](https://github.com/vllm-project/vllm/pull/32603) |
| 作者 | lingebeng |
| 合并日期 | 2026-01-20 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
测试工具 `_num_tokens_to_min_blocks` 用 `(n+b)//b` 计算最小块数,当 token 数恰为块大小整数倍时会多算 1 块;改为标准向上取整 `(n+b-1)//b`。

## 问题现象
正文只有一句话:"Fix Off-by-one error in _num_tokens_to_min_blocks calculation",并附加说明 "Maybe on purpose, need review."(作者自己也表示不确定原写法是否有意为之,需要 review),未描述具体故障现象。问题位于 `tests/kernels/utils.py` 的 `_num_tokens_to_min_blocks`:该函数按 docstring 语义计算「容纳 num_tokens 个 token 所需的最少 KV cache 块数」,但原公式总是比正确的向上取整多出 1 块;同文件 `make_block_tables_slot_mapping` 的 docstring 中给出的公式说明也一并修正。

## 根因
正文未披露根因分析,仅定性为 off-by-one。(由 diff 推断)原实现 `return (num_tokens + block_size) // block_size` 在整数除法下恒等于 `num_tokens // block_size + 1`:当 `num_tokens` 不是 `block_size` 整数倍时恰为向上取整,但当 `num_tokens` 恰好是整数倍时(如 16 个 token、block_size 16),应为 1 块却算出 2 块,即多算一块;正确的最小块数应为向上取整 `(num_tokens + block_size - 1) // block_size`。

## 修复方案
改动仅 `tests/kernels/utils.py` 两处:

```diff
-    return (num_tokens + block_size) // block_size
+    return (num_tokens + block_size - 1) // block_size
```

一是 `_num_tokens_to_min_blocks` 的返回公式,二是 `make_block_tables_slot_mapping` docstring 中的公式说明同步改为 `num_blocks = (num_tokens + block_size - 1) // block_size`,保持文档与实现一致。

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:`tests/kernels/utils.py` 是内核测试辅助代码,`_num_tokens_to_min_blocks` 与 `make_block_tables_slot_mapping` 为使用它的测试生成最小 block tables/slot mapping;修正后这些测试在 num_tokens 为块大小整数倍的场景下所需块数减 1,依赖该辅助函数的测试用例的预期值会随之变化。作者注明「Maybe on purpose」,提示原公式可能被某些测试有意依赖,合并前经过 review。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
