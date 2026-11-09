# PR #21189: Revert "[bugfix] Fix PPMissingLayer AttributeError when Using PP"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21189](https://github.com/sgl-project/sglang/pull/21189) |
| 作者 | ch-wan |
| 合并日期 | 2026-03-23 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #19804:移除 PPMissingLayer 上返回 self 的 `__getattr__` 与静默吞赋值的 `__setattr__` 覆写,恢复 nn.Identity 默认行为。

## 问题现象
正文未描述具体现象,仅一句话说明回滚 #19804([bugfix] Fix PPMissingLayer AttributeError when Using PP)。没有列出报错堆栈、复现脚本或受影响配置。

## 根因
正文未披露根因。从 diff 看,#19804 曾给 `PPMissingLayer`(流水线并行下缺失层的占位符,继承 `torch.nn.Identity`)添加两个覆写(机制描述由 diff 推断):

- `__getattr__`:常规属性查找失败(AttributeError/KeyError)时返回 `self`,让 `isinstance(layer.mlp, SomeMoE)` 之类判断安全为 False;
- `__setattr__`:除 `return_tuple` 与下划线开头外的属性赋值被静默丢弃。

本回滚将两者删除。被静默吞掉的赋值与永远返回 self 的属性访问是行为相当激进的全局语义,正文未说明它引发了什么新问题,仅以回滚处置。

## 修复方案
`python/sglang/srt/layers/utils/common.py` 的 `PPMissingLayer` 删除 `__getattr__` 与 `__setattr__` 两个覆写,仅保留 `__init__`(设置 `return_tuple`)与原 `forward`;类文档恢复为原注释(不含 #19804 添加的说明段):

```diff
-    def __getattr__(self, name: str):
-        try:
-            return super().__getattr__(name)
-        except (AttributeError, KeyError):
-            return self
-
-    def __setattr__(self, name: str, value):
-        if name == "return_tuple" or name.startswith("_"):
-            super().__setattr__(name, value)
-        else:
-            pass
-
```

## 影响与回归风险
- 影响窗口:正文未说明(#19804 合并至本回滚之间)。
- 回归面:流水线并行(PP)下 `PPMissingLayer` 的属性访问/赋值行为回到 #19804 之前——对占位符访问不存在属性会重新抛 AttributeError;#19804 想修的 PP AttributeError 若真实存在则回归(正文未评价);#19804 期间依赖「属性访问返回 self」语义的代码路径需要重新审视。

## 关联
- [#19804](https://github.com/sgl-project/sglang/pull/19804)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
