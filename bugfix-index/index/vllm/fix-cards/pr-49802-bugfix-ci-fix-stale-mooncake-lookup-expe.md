# PR #49802: [Bugfix][CI] Fix stale Mooncake lookup expectation broken by a merge race

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49802](https://github.com/vllm-project/vllm/pull/49802) |
| 作者 | hmellor |
| 合并日期 | 2026-07-25 |
| labels | bug, kv-connector, ready, v1 |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
两个 Mooncake PR 合并竞争让 main 留下互相矛盾的 lookup 期望值;把测试请求改为 33 token 使 hit 停在请求末尾之下。

## 问题现象
`tests/v1/kv_connector/unit/test_mooncake_store_worker.py::test_lookup_rejects_boundary_missing_one_mamba_shard` 在 `main` 上失败,导致每个无关 PR 的 `V1 Core + KV + Metrics` CI 步骤全挂(如 buildkite build 80149)。失败断言:

```
AssertionError: assert 16 == 32
  where 16 = lookup(32, [b'h0', b'h1'])
```

## 根因
正文披露:这是两个 Mooncake PR 的合并竞争(merge race),不是功能性回归。#49481(a76df87)修改 `MooncakeStoreWorker.lookup`,使覆盖整个请求的 hit 会在请求末尾之下重新推导(re-derived),保留最后一个 token 用于采样——`block_size=16` 时,完整的 2-block hit 因此返回 16,且 #49481 新增的 `test_lookup_full_hit_reuses_existing_boundary` 断言的正是 `lookup(32, [b"h0", b"h1"]) == 16`。而 #49499(70052fb)从早于 #49481 约 3 小时的 `f83de6d` 分出,其新测试断言同一调用返回 32;该 PR 在旧基线上 CI 全绿、无文本冲突,一天后未 rebase 就合并,使 `main` 上对同一调用留下两个互相矛盾的期望。两个 PR 的生产代码都正确,只有测试期望过时。

## 修复方案
只改 `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` 的 `test_lookup_rejects_boundary_missing_one_mamba_shard`:请求 token 数从 32 改为 33,使 hit 停在请求末尾之下、不触发 full-hit 重新推导,保住测试原意(所有 shard 都在 ⇒ 命中 32;缺一个 Mamba shard ⇒ 0),而不是把断言弱化成 `== 16`;同时与 #49481 在文件中其他地方使用的 `lookup(65, ...) == 64` 惯用法一致。

```diff
-    assert worker.lookup(32, [b"h0", b"h1"]) == 32
+    assert worker.lookup(33, [b"h0", b"h1"]) == 32
```

验证:main @ dbcc1cdd0a 上该测试失败(`assert 16 == 32`);修复后整个文件 84 passed(19.5s),包括断言相反值的 #49481 测试。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号;自两个 PR(#49481/#49499)先后合并且未 rebase 起,至本修复前。
- 回归面:仅单元测试文件,不触及运行时代码,服务行为与精度不受影响(正文明确说明)。

## 关联
- #49481(改变 lookup 语义的 PR)
- #49499(带过时期望的 PR,其所修 bug 的判别力在本修复中保留)
- [buildkite build 80149](https://buildkite.com/vllm/ci/builds/80149)(失败示例)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
