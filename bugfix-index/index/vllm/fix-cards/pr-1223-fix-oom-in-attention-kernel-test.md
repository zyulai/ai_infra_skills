# PR #1223: Fix OOM in attention kernel test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#1223](https://github.com/vllm-project/vllm/pull/1223) |
| 作者 | WoosukKwon |
| 合并日期 | 2023-09-28 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
attention kernel 测试用 40K 超长序列致参考实现 OOM,测试内把采样上限压到 4096 修复。

## 问题现象
`tests/kernels/test_attention.py` 的 prompt attention 测试此前使用非常大的序列长度(如 40K),导致不使用 flash attention 的参考实现(reference implementation)发生 OOM。

## 根因
正文披露:测试选用的超大序列长度使参考实现显存超限;作者认为 xformers 已有自己的测试覆盖,因此在本测试中改用更短序列是可以接受的。

## 修复方案
`test_multi_query_kv_attention` 中对采样上限取 min:

```diff
-    seq_lens = random.sample(range(1, MAX_SEQ_LEN), num_seqs)
-    seq_lens[-1] = MAX_SEQ_LEN
+    # MAX_SEQ_LEN sometimes causes OOM in the reference implementation.
+    # As the xformers library is already tested with its own tests, we can use
+    # a smaller MAX_SEQ_LEN here.
+    max_len = min(MAX_SEQ_LEN, 4096)
+    seq_lens = random.sample(range(1, max_len), num_seqs)
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:仅测试代码;该测试覆盖的最大序列长度降到 4096,对被测 vLLM attention kernel 的极端长度覆盖减弱(正文以此为可接受取舍),不触及任何产品代码。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
