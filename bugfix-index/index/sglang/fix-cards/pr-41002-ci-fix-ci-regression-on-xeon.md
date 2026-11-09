# PR #41002: [CI] fix CI regression on xeon

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#41002](https://github.com/sgl-project/sglang/pull/41002) |
| 作者 | xinguozhu-2026 |
| 合并日期 | 2026-09-28 |
| labels | intel, cpu, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

修 xeon CI 两处:MiniMax-H3 测试的 patch 目标路径过时而失效,`test_hash_topk` 的 EP=4 mock 缺 `tp_size=4`/`attn_tp_size=4`。

## 问题现象

两处 xeon(CPU)CI 失败:

1. MiniMax-H3 CI 回归,失败用例 `test_subblock_sparse_attention.py::TestSubBlockSparseAttentionModalities`,原始日志见 Actions run 35834418761/job/107094633286(?pr=40139)。
2. `test_hash_topk.py` 中存在错误的测试用例,原始日志见 Actions run 35946733998/job/107466322671(?pr=41002)。

## 根因

正文对两处问题分别说明:

1. MiniMax-H3 回归:正文未直接披露代码层根因。(由 diff 推断)测试中被 `patch` 的目标模块 `sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_global_forced_attn_backend` / `get_attn_backend` 已不存在(相关实现位于 `sglang.multimodal_gen.runtime.layers.attention.selector`),patch 路径过时导致测试失败。
2. `test_hash_topk`:正文明确写 "EP = 4 fiction needs `tp_size = 4` same the `attn_tp_size`"——测试模拟 moe_ep_size=4 时必须同时给 `tp_size=4` 与 `attn_tp_size=4`,原用例缺失这两个 override,属测试用例本身写错。

## 修复方案

改两个测试文件(均为测试代码,不触及运行时逻辑):

`test/registered/cpu/test_subblock_sparse_attention.py`:8 处 `patch` 目标从 `sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.` 前缀改为 `sglang.multimodal_gen.runtime.layers.attention.selector.` 前缀(涉及 `get_global_forced_attn_backend` 与 `get_attn_backend`):

```diff
-                "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3."
+                "sglang.multimodal_gen.runtime.layers.attention.selector."
                 "get_global_forced_attn_backend",
```

`test/registered/moe/test_hash_topk.py`:

```diff
-        get_parallel().override(moe_ep_size=4, moe_ep_rank=2),
+        get_parallel().override(
+            tp_size=4, attn_tp_size=4, moe_ep_size=4, moe_ep_rank=2
+        ),
```

## 影响与回归风险

- 影响窗口:正文未说明引入回归的具体提交;受影响的是 xeon CPU CI 上这两个测试文件。
- 回归面:仅测试代码:patch 目标路径与 parallel override 参数。不触及推理/模型代码,线上行为无影响;若后续相关函数再次迁移,patch 路径需同步更新。

## 影响验证

正文给出本地验证:`pytest test/registered/cpu/test_subblock_sparse_attention.py -q` → 31 passed, 3 warnings, 7 subtests passed in 36.28s;`pytest -q test/registered/moe/test_hash_topk.py` → 3 passed, 4 warnings in 15.53s。

## 关联

- MiniMax-H3 失败日志:[actions/runs/35834418761/job/107094633286](https://github.com/sgl-project/sglang/actions/runs/35834418761/job/107094633286?pr=40139)(关联 pr=40139)
- test_hash_topk 失败日志:[actions/runs/35946733998/job/107466322671](https://github.com/sgl-project/sglang/actions/runs/35946733998/job/107466322671?pr=41002)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
