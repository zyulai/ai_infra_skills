# PR #1481: fix incorrect links in documentation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1481](https://github.com/sgl-project/sglang/pull/1481) |
| 作者 | rchen19 |
| 合并日期 | 2024-09-21 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
文档多处链接写成仓库内文件路径,readthedocs 上点击无法跳转;改为指向 readthedocs / GitHub 的完整 URL。

## 问题现象
当前文档站点(sglang.readthedocs.io/en/latest)上部分链接实际是仓库内的文件路径,例如 `https://sglang.readthedocs.io/en/latest/backend.html#docs/en/custom_chat_template.md`,本应指向仓库文件 `docs/en/custom_chat_template.md`,但在网站上点击该链接不会跳转到任何地方。

## 根因
正文说明:这些链接写成了仓库内文件路径而非完整 URL,渲染到文档站点后成为无效链接(正文未进一步披露更深层的构建机制)。

## 修复方案
修改 `docs/en/backend.md`、`docs/en/frontend.md`、`docs/en/model_support.md` 三个文件,把含文件路径的链接改为完整 URL:
- 文档页指向 readthedocs 页面,如 `docs/en/custom_chat_template.md` → `https://sglang.readthedocs.io/en/latest/custom_chat_template.html`(sampling_params、hyperparameter_tuning、model_support 同理);
- 仓库文件与示例目录指向 GitHub `main` 分支的 blob/tree URL,如 `test/srt/test_vision_openai_server.py`、`examples/frontend_language/usage/readme_examples.py` 等;
- 顺带将 model_support.md 中指向 `llama2.py` 的示例链接修正为 `llama.py`,并把 README 的相对链接改为完整 GitHub URL。

```markdown
-Learn more about the argument format [here](docs/en/sampling_params.md).
+Learn more about the argument format `here <https://sglang.readthedocs.io/en/latest/sampling_params.html>`_.
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:纯文档改动,不涉及任何代码与运行路径;风险仅在于链接目标将来页面迁移时可能再次失效。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
