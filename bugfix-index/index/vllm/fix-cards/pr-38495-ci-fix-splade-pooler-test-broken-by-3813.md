# PR #38495: [CI] Fix SPLADE pooler test broken by #38139

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38495](https://github.com/vllm-project/vllm/pull/38495) |
| 作者 | haosdent |
| 合并日期 | 2026-03-30 |
| labels | ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#38139 给 PoolingMetadata 新增 get_prompt_token_ids_cpu(),SPLADE 测试的 SimpleNamespace mock 缺该方法抛 AttributeError;改用真实 dataclass 修复。

## 问题现象
SPLADE pooler 测试 `test_splade_pooler_matches_reference_formula` 失败,报 `AttributeError`:测试用 `types.SimpleNamespace` 伪造的元数据对象只有 `prompt_lens` 与 `prompt_token_ids` 两个属性,而 #38139 给 `PoolingMetadata` 新增的 `get_prompt_token_ids_cpu()` 在其上不存在。验证命令:`pytest tests/models/language/pooling/test_splade_sparse_pooler.py -xvs` 本地通过。

## 根因
正文披露:PR #38139 向 `PoolingMetadata` 添加了 `get_prompt_token_ids_cpu()` 方法,但测试里以 `SimpleNamespace` 模拟的 `PoolingMetadata` 没有该方法,调用时抛 `AttributeError`;正文同时指出,继续用 mock 会在每次接口变更时重复坏掉,应改用真实 dataclass。

## 修复方案
`tests/models/language/pooling/test_splade_sparse_pooler.py` 中,把 `types.SimpleNamespace(prompt_lens=..., prompt_token_ids=...)` 替换为真实构造的 `PoolingMetadata`(含 `prompt_token_ids_cpu=token_ids`、`pooling_params=[PoolingParams(task="embed")] * B`、`pooling_states=[PoolingStates() for _ in range(B)]`):

```diff
-    meta = types.SimpleNamespace(
-        prompt_lens=prompt_lens_tenser, prompt_token_ids=token_ids
+    meta = PoolingMetadata(
+        prompt_lens=prompt_lens_tenser,
+        prompt_token_ids=token_ids,
+        prompt_token_ids_cpu=token_ids,
+        pooling_params=[PoolingParams(task="embed")] * B,
+        pooling_states=[PoolingStates() for _ in range(B)],
     )
```

本 PR 取代(supersedes)此前的回退 PR #38490。

## 影响与回归风险
- 影响窗口:#38139 合并之后至本 PR 之前,`test_splade_sparse_pooler.py` 一直处于损坏状态。
- 回归面:纯测试改动,不触及生产代码。改用真实 `PoolingMetadata` 后,未来 `PoolingMetadata` 接口再变更会在构造处直接显形,而不是等运行到缺失方法时才报错;代价是测试需随 dataclass 字段同步维护。

## 关联
- PR #38139(新增方法、触发回归)
- PR #38490(被本 PR 取代的回退方案)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
