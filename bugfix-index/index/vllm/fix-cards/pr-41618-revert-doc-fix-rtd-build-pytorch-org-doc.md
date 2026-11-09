# PR #41618: Revert "[Doc] Fix RTD build: pytorch.org/docs/stable/objects.inv returns 404"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41618](https://github.com/vllm-project/vllm/pull/41618) |
| 作者 | stecasta |
| 合并日期 | 2026-05-04 |
| labels | ready |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
pytorch.org stable 的 objects.inv 已在上游恢复,回滚临时 2.11 版本 pin,mkdocs 交叉引用回到 stable。

## 问题现象
此前因 `https://pytorch.org/docs/stable/objects.inv` 返回 404 导致 ReadTheDocs 构建(#41353)把 mkdocs.yaml 临时 pin 到 `/docs/2.11/objects.inv` 并加了 TODO 注释;现上游已恢复,该临时方案需要回退。正文未描述运行时现象,属文档构建配置问题。

## 根因
正文披露:404 问题已被上游修复(pytorch/docs#85),`pytorch.org/docs/stable/objects.inv` 重新可用,且与 `/docs/2.11/` pin 的文件 byte-identical(逐字节一致),因此临时 pin 不再必要,应恢复指向 stable。

## 修复方案
直接 revert #41353,仅改 `mkdocs.yaml` 一处:pytorch objects.inv URL 从 `https://pytorch.org/docs/2.11/objects.inv` 恢复为 `https://pytorch.org/docs/stable/objects.inv`,并删除指向 pytorch/pytorch#182007 的 TODO 注释。

关键 diff 摘录:
```diff
-          # TODO revert to stable once https://github.com/pytorch/pytorch/issues/182007 is fixed
-          - https://pytorch.org/docs/2.11/objects.inv
+          - https://pytorch.org/docs/stable/objects.inv
```

## 影响与回归风险
- 影响窗口:#41353 合并之后。
- 回归面:仅文档构建配置(mkdocs 的 inventory URL),不影响任何运行时代码路径;若上游 stable objects.inv 再次 404,RTD 构建可能重现同样的交叉引用失败。

## 关联
- Reverts [vllm-project/vllm#41353](https://github.com/vllm-project/vllm/pull/41353)
- 上游修复 [pytorch/docs#85](https://github.com/pytorch/docs/pull/85)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
