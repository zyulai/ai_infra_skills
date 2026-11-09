# PR #20185: [BugFix] Fix the incorrect func name in the comments. (config.py)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20185](https://github.com/vllm-project/vllm/pull/20185) |
| 作者 | 1195343015 |
| 合并日期 | 2025-06-28 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
config.py 注释里的函数名写错,把 check_and_update_configs() 更正为 check_and_update_config(),纯文档修正。

## 问题现象
正文未描述具体现象。这是文档注释层面的错误:`vllm/config.py` 中 `CacheConfig` 的 docstring(围绕 block size 的说明)引用的函数名与实际函数不一致,会误导开发者按注释去查找配置更新逻辑;对运行时行为无影响。

## 根因
正文披露:注释中的函数名不正确(Purpose 原文为「Fix the incorrect func name in the comments. (config.py)」)。(由 diff 推断)注释把 `Platform.check_and_update_config()` 误写成了复数形式 `Platform.check_and_update_configs()`,代码库中并不存在后者。

## 修复方案
仅改 `vllm/config.py` 中 `CacheConfig` docstring 的一行,把注释中的函数名改为真实存在的函数:

```diff
-    be set in `Platform.check_and_update_configs()` based on the current
+    be set in `Platform.check_and_update_config()` based on the current
      platform."""
```

无任何逻辑代码改动。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:无运行时代码改动,仅 docstring 文本变更,不影响任何功能路径与配置行为。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
