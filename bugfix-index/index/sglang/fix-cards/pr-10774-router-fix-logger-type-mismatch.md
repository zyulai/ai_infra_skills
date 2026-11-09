# PR #10774: [router] fix logger type mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10774](https://github.com/sgl-project/sglang/pull/10774) |
| 作者 | CatherineSue |
| 合并日期 | 2025-09-23 |
| labels | bug, router, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
router 日志级别选项用 `warning` 而 Rust 端期望 `warn`,取值不匹配;CLI choices 与文档统一改为 `warn`。

## 问题现象
正文未描述具体现象。按正文所述:logger 的取值需从 `warning` 改为 `warn`,因为 Rust 端期望 `warn` 而非 `warning`,即 Python 侧暴露的日志级别取值与 Rust 路由器实际接受的不一致。

## 根因
正文披露:Rust 端期望 `warn` 而不是 `warning`,而 Python 侧的日志级别选项列表里写的是 `warning`(以及 `critical`),两边取值不匹配。

## 修复方案
- `sgl-router/py_src/sglang_router/router_args.py`:`--log-level` 的 argparse `choices` 由 `["debug", "info", "warning", "error", "critical"]` 改为 `["debug", "info", "warn", "error"]`;
- `sgl-router/py_src/sglang_router/router.py`:同步更新 `log_level` 参数的 docstring(Options: 'debug', 'info', 'warn', 'error')。

```python
-            choices=["debug", "info", "warning", "error", "critical"],
+            choices=["debug", "info", "warn", "error"],
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 sgl-router 的 CLI 参数校验与文档字符串;取值集合收窄,此前传 `warning` 或 `critical` 的调用/脚本将不再通过 argparse 校验,需改用 `warn`(或 `error`)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
