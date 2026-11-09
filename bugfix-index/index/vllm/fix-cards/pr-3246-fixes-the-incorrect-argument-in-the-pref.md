# PR #3246: Fixes the incorrect argument in the prefix-prefill test cases

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3246](https://github.com/vllm-project/vllm/pull/3246) |
| 作者 | sighingnow |
| 合并日期 | 2024-03-16 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`tests/kernels/test_prefix_prefill.py` 的 `num_queries_per_kv` 参数化误用 `NUM_HEADS` 列表,改为正确的 `NUM_QUERIES_PER_KV`,并顺手修了参考输出比较与多 GPU 挂起问题。

## 问题现象
正文仅写 "See also comment in https://github.com/vllm-project/vllm/pull/3007",未描述具体现象。问题位于 `tests/kernels/test_prefix_prefill.py` 的 `test_contexted_kv_attention`:该测试用 `@pytest.mark.parametrize("num_queries_per_kv", NUM_HEADS)` 做参数化,而按测试矩阵语义此处应枚举每 KV 头对应的 query 头数(`NUM_QUERIES_PER_KV`),用 `NUM_HEADS` 是错误实参,导致该维度没有按预期取值组合被覆盖/比较。

## 根因
正文未披露根因,只指向 PR #3007 中的评论。(由 diff 推断)参数化列表写错常量:`num_queries_per_kv` 维度被喂了 `NUM_HEADS`,属于测试编写时的笔误;此外参考输出比较用 `output_ref.squeeze(0, 2)` 与实际输出形状不匹配(改为按 `output.shape` reshape),多 GPU 环境下未显式 `torch.cuda.set_device(device)` 会导致 capture graph 时进程同时跑在 GPU0/GPU1 上挂起(diff 注释引用 flash-attention issue #523 的同类问题)。

## 修复方案
`tests/kernels/test_prefix_prefill.py` 三处:
1. 参数化实参修正:`@pytest.mark.parametrize("num_queries_per_kv", NUM_HEADS)` 改为 `NUM_QUERIES_PER_KV`;
2. 测试开头加 `torch.cuda.set_device(device)`,附注释说明否则多 GPU capture graph 时会挂起(参见 https://github.com/Dao-AILab/flash-attention/issues/523 );
3. 参考输出比较:`output_ref.squeeze(0, 2)` 改为 `output_ref.reshape(output.shape)`,再与 `output` 做 `torch.allclose`(atol=1e-6)。

```diff
-@pytest.mark.parametrize("num_queries_per_kv", NUM_HEADS)
+@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:仅 `tests/kernels/test_prefix_prefill.py` 测试文件,不触及任何生产代码;参数化修正后该测试将按 `NUM_QUERIES_PER_KV` 的取值枚举,测试覆盖组合发生变化(此前重复/错位的组合被替换);`reshape(output.shape)` 使参考输出与实际输出形状严格一致,比较更严谨。

## 关联
- See also: https://github.com/vllm-project/vllm/pull/3007 (评论)
- diff 注释引用: https://github.com/Dao-AILab/flash-attention/issues/523

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
