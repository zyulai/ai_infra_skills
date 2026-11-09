# PR #20315: fix[Docs]: link anchor is incorrect #20309

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20315](https://github.com/vllm-project/vllm/pull/20315) |
| 作者 | yyzxw |
| 合并日期 | 2025-07-02 |
| labels | documentation, structured-output, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
多篇文档引用的 OpenAI-Compatible Server 锚点失效,统一改用 `serving-openai-compatible-server` 修复死链。

## 问题现象
正文以两段录屏(before/after)展示:文档中指向 "OpenAI-Compatible Server" 页面的交叉链接点击后无法正确跳转(链接失效);修复后可正常跳转。正文未以文字描述更多症状,也未说明受影响的具体文档版本。

## 根因
正文仅写 "fix error link",未披露根因。(由 diff 推断)文档交叉引用所用的锚点标识与目标页面实际定义的锚点不一致,导致 mkdocs 交叉引用解析失败、链接 404;本次将目标页(docs/serving/openai_compatible_server.md 顶部)的锚点定义由 `[](){ #openai-compatible-server }` 更名为 `[](){ #serving-openai-compatible-server }`,并把全部引用同步更新为该名称。

## 修复方案
共修改 8 个文档文件,统一锚点名称:
- docs/configuration/engine_args.md、docs/design/arch_overview.md、docs/features/structured_outputs.md、docs/getting_started/installation/intel_gaudi.md、docs/models/generative_models.md、docs/models/pooling_models.md、docs/models/supported_models.md:将 `[...][openai-compatible-server]` 形式的引用改为 `[...][serving-openai-compatible-server]`;
- docs/serving/openai_compatible_server.md:锚点定义改名。

```diff
-[](){ #openai-compatible-server }
+[](){ #serving-openai-compatible-server }
```

## 影响与回归风险
- 影响窗口:正文未说明自哪个版本起链接失效,不予推断。
- 回归面:纯文档改动,不涉及任何运行时代码;风险仅在于若有外部文档或脚本按旧锚点 `#openai-compatible-server` 直链,改名后仍会指向旧地址。仓库内引用已在本次全部同步。

## 关联
- https://github.com/vllm-project/vllm/issues/20309

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
