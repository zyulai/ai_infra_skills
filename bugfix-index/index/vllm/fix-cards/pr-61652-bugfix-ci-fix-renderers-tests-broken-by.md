# PR #61652: [Bugfix][CI] Fix renderers tests broken by tool name removal from warn logs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61652](https://github.com/vllm-project/vllm/pull/61652) |
| 作者 | ganeshr10 |
| 合并日期 | 2026-10-14 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#59294 删除警告日志中的工具名但未同步测试,7 个 renderers 测试在 main 持续红,本 PR 修正测试断言恢复绿色。

## 问题现象
自 #59294 合并起,`tests/renderers/test_chat_utils.py` 中 7 个测试在 main 上持续失败,每个触及 `cpu-async-engine-deps` 路径的 PR 都会撞上。失败形态一致,例如:
`FAILED test_tool_call_arguments_malformed_json_small`,断言 `assert 'exec' in 'Tool call has arguments that are not valid JSON (54 chars); coercing to an empty object so the conversation can continue.'`。#59294 自己的 `:computer: (CPU) Reasoning + Renderers` job(build 94769)也因同样原因失败,但被红着合并。

## 根因
正文披露:#59294 移除了 `vllm/renderers/chat_utils/chat_message.py` 中两条 malformed-`arguments` 警告里的 tool name,但没有同步更新 `tests/renderers/test_chat_utils.py`,测试仍断言工具名会出现在发出的日志记录里。

## 修复方案
只改测试,源码不动(#59294 不打工具名是有意行为):删除 4 处工具名断言,保留消息中仍存在的部分断言。工具名原本是 large 与 small 用例间的唯一区别,故 large 用例改为断言按长度记录(18 KB payload 被摘要为长度而非全文记录):

```diff
-    assert "write" in caplog.records[0].message
     assert "coercing to an empty object" in caplog.records[0].message
+    assert f"{len(bad)} chars" in caplog.records[0].message
```

验证:`pytest -q tests/renderers/test_chat_utils.py -k tool_call_arguments` → 12 passed。

## 影响与回归风险
- 影响窗口:正文说明自 #59294 合并起 main 上一直失败。
- 回归面:仅 `tests/renderers/test_chat_utils.py` 的断言变化,不触及运行时代码;large 用例的覆盖点从"含工具名"改为"按长度摘要"。

## 关联
[#59294](https://github.com/vllm-project/vllm/pull/59294)、[build 94769](https://buildkite.com/vllm/ci/builds/94769)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
