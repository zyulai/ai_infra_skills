# PR #66044: [Doc][Bugfix] Fix strict docs build broken by unresolved autorefs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66044](https://github.com/vllm-project/vllm/pull/66044) |
| 作者 | shen-shanshan |
| 合并日期 | 2026-11-06 |
| labels | bug, minimax, nvidia |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
docstring 中连续方括号被当引用式链接、以及类丢失 docstring,导致 strict MkDocs 构建 12 个 autorefs 警告中止;分别改为代码 span 和补 docstring。

## 问题现象
strict MkDocs 构建(Read the Docs,`fail_on_warning: true`)自 2026-11-05 起在 `main` 上持续失败,报 `Aborted with 12 warnings in strict mode!`,所有 PR 的 docs check 都失败(如 Read the Docs build 35688234,`main` at `e336407a3d`)。影响面:整个仓库的文档 CI 阻塞。

## 根因
正文披露两个根因:
1. MiniMax-M3.1 drafter 的 docstring(#65386 引入,11 个警告):部分 docstring 用连续方括号写数组形状,如 `red[8 t + wave][row][token]`、`x [16][K]`、`x_in [rows][6144]` 等;Markdown 把 `[a][b]` 解析为引用式链接,autorefs 尝试解析 `b`(`row`、`K`、`6144` 等标签)失败。
2. `TrtLlmMxfp4ExpertsModular` 丢失 docstring(#54109 引入,1 个警告):#54109 把原被文档化的类改名为 `TrtLlmMxfp4ExpertsModularBase`,新增的 `TrtLlmMxfp4ExpertsModular` 与 `TrtLlmMxfp4Mxfp8ExpertsModular` 子类没有 docstring;mkdocstrings 默认跳过无 docstring 对象,API 页面没有对应锚点,`docs/design/moe_kernel_features.md` 中的链接无法解析。

## 修复方案
- 把 `vllm/models/minimax_m3/amd/mono/drafter/device.py`(`red_store`、`xq_chunk`、`row_x`、`quant_store`)与 `drafter/attention.py`(`store_partial`)中连续方括号的形状写法包进代码 span:

```diff
-    """Wave ``wave``'s partial of tile ``t`` -> red[8 t + wave][row][token]."""
+    """Wave ``wave``'s partial of tile ``t`` -> ``red[8 t + wave][row][token]``."""
```

单层方括号(如 `[1, rank, site]`)不受影响,保持原样。
- 给 `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py` 的两个子类各加一行 docstring(说明其激活量化方式),使其获得 API 锚点;`moe_kernel_features.md` 的链接保持不变(它指向具体 experts 类)。
仅改 docstring,无代码行为变化。作者在与 Read the Docs 相同命令的干净 venv 中验证:`main` Exit 1(同样 12 个警告),本 PR Exit 0、无警告。

## 影响与回归风险
- 影响窗口:2026-11-05(#65386/#54109 相关改动进入 main)至本 PR 之间的 strict docs 构建失败。
- 回归面:仅 docstring 文本;无代码行为变化。补上的 docstring 会让两个 TRTLLM MXFP4 experts 子类出现在 API 文档中。

## 关联
- #65386(连续方括号 docstring 来源)
- #54109(类改名/子类无 docstring 来源)
- Read the Docs build 35688234、35688313(正文链接的失败构建)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
