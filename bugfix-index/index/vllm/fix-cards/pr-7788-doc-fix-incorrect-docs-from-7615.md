# PR #7788: [Doc] Fix incorrect docs from #7615

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7788](https://github.com/vllm-project/vllm/pull/7788) |
| 作者 | petersalas |
| 合并日期 | 2024-08-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
修正 #7615 遗留的文档错误:supported_models.rst 里 :code: 后多余空格,音频示例 docstring 误写成 vision language models。

## 问题现象
正文说明:#7615 在文档中留下了错误的格式化和注释里的错误复制粘贴。具体位置由 diff 可见:1) `docs/source/models/supported_models.rst` 多模态模型表格中,`UltravoxModel` 与 `fixie-ai/ultravox-v0_3` 两处写成 `:code: `(冒号后多一个空格)(由 diff 推断:该写法在 RST 中不再是合法的 `:code:` 角色,对应表格单元格无法正确渲染);2) `examples/offline_inference_audio_language.py` 文件头 docstring 声称该示例演示 "vision language models"(视觉语言模型)的 prompt 格式,而它实际是音频语言模型示例。

## 根因
正文披露:这些错误由 #7615 引入——该 PR 在文档中造成格式错误和注释中的错误复制粘贴。

## 修复方案
改动两个文件:
1. `docs/source/models/supported_models.rst`:去掉 UltravoxModel 行两处 `:code: ` 中多余的空格。关键 diff 摘录:

```diff
-  * - :code: `UltravoxModel`
+  * - :code:`UltravoxModel`
     - Ultravox
     - Audio
-    - :code: `fixie-ai/ultravox-v0_3`
+    - :code:`fixie-ai/ultravox-v0_3`
```

2. `examples/offline_inference_audio_language.py`:docstring 中 `with the correct prompt format on vision language models.` 改为 `...on audio language models.`,与示例实际内容一致。

## 影响与回归风险
- 影响窗口:自 #7615 合入引入错误起,至本修复(正文未给出版本号)。
- 回归面:纯文档(RST 表格)与示例脚本 docstring 文本,不涉及任何运行时代码,回归风险极低。

## 关联
- https://github.com/vllm-project/vllm/pull/7615

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
