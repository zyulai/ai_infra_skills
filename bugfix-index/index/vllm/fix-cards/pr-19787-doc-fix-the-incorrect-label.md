# PR #19787: [doc] fix the incorrect label

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19787](https://github.com/vllm-project/vllm/pull/19787) |
| 作者 | reidliu41 |
| 合并日期 | 2025-06-18 |
| labels | documentation, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
CUDA 安装文档中 `--8<--` 片段引用的标签写错,指向不存在的 `create-a-new-python-environment`;改为正确的 `set-up-using-python` 标签并删掉多余小节标题。

## 问题现象
正文未用文字描述具体现象,仅给出 From/To 两张截图对比说明文档显示不正确,涉及 `docs/getting_started/installation/gpu/cuda.inc.md` 相关内容;修复后 gpu.md 中 NVIDIA CUDA 安装小节应引用正确的 `set-up-using-python` 片段。

## 根因
正文明确指出标签写错:「It should `set-up-using-python` in `docs/getting_started/installation/gpu/cuda.inc.md`」,即 gpu.md 中引用的标签应为 `set-up-using-python` 而非实际写成的标签。
(由 diff 推断)diff 中 gpu.md 原引用为 `docs/getting_started/installation/gpu/cuda.inc.md:create-a-new-python-environment`,而 cuda.inc.md 是用 `# --8<-- [start:set-up-using-python]` 区域标记定义片段的,`create-a-new-python-environment` 并非有效的片段标签,导致该 `--8<--` 指令无法取到预期片段内容。

## 修复方案
两处改动:
1. `docs/getting_started/installation/gpu.md`:NVIDIA CUDA 小节的 `--8<--` 引用从 `...cuda.inc.md:create-a-new-python-environment` 改为 `...cuda.inc.md:set-up-using-python`。
2. `docs/getting_started/installation/gpu/cuda.inc.md`:删除多余的 `### Create a new Python environment` 小节标题(该片段由区域标记圈定,标题不再需要)。

```diff
-    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:create-a-new-python-environment"
+    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:set-up-using-python"
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅文档文件改动,不影响任何运行时代码。受影响的是 `docs/getting_started/installation/gpu.md` 中 NVIDIA CUDA 小节的渲染内容;同时 cuda.inc.md 删除标题会改变该 include 文件自身的文档结构,其他引用同一文件标签的页面不受影响(本次未改动任何区域标记)。

## 关联
正文未列出(仅附截图链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
