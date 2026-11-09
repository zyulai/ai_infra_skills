# PR #19586: [doc] fix incorrect link

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19586](https://github.com/vllm-project/vllm/pull/19586) |
| 作者 | reidliu41 |
| 合并日期 | 2025-06-13 |
| labels | documentation, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
docs/design/multiprocessing.md 日志示例中的排障文档链接指向旧路径,修正为 `usage/troubleshooting.html` 对应锚点。

## 问题现象
正文仅写「fix the link」,未描述其他现象。问题位于 `docs/design/multiprocessing.md`:该页在讲解多进程行为时展示了一段真实的 vLLM WARNING 日志示例("CUDA was previously initialized. We must use the `spawn` multiprocessing start method. Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'"),日志文本末尾附带排障文档链接 `https://docs.vllm.ai/en/latest/usage/debugging.html#python-multiprocessing`。由于 vLLM 实际运行时打印的就是这条日志,读者照着日志里的 URL 去查阅 "#python-multiprocessing" 排障说明时会落到错误的页面路径,找不到对应内容。属纯文档问题,不影响运行时行为,影响面为阅读该文档的全部用户。

## 根因
正文未披露根因,只说明目的是「fix the link」。(由 diff 推断)文档中引用的 URL 路径 `usage/debugging.html` 已不是有效路径,排障内容实际位于 `usage/troubleshooting.html`;推测站点文档结构调整后该处引用没有同步更新,锚点 `#python-multiprocessing` 仍挂在旧路径下。

## 修复方案
仅改动 `docs/design/multiprocessing.md` 一处(日志示例内):把链接由 `https://docs.vllm.ai/en/latest/usage/debugging.html#python-multiprocessing` 改为 `https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing`,其余内容不变。

```diff
-    https://docs.vllm.ai/en/latest/usage/debugging.html#python-multiprocessing
+    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:纯文档改动,不触及任何代码路径,无运行时回归风险;仅影响阅读该文档并点击该链接的读者体验。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
