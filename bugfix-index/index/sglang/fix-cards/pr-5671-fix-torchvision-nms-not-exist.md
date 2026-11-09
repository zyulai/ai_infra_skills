# PR #5671: fix torchvision::nms not exist

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5671](https://github.com/sgl-project/sglang/pull/5671) |
| 作者 | zhyncs |
| 合并日期 | 2025-04-23 |
| labels | (none) |
| 热度 | 👍 8 · 💬 0 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
CI 的 sgl-kernel 测试环境未装 torchvision,补装 torchvision 以消除 torchvision::nms 报错。

## 问题现象
正文未描述具体现象(Motivation/Modifications 均为空模板),从标题看是 `torchvision::nms not exist` 报错。改动仅涉及 `.github/workflows/pr-test-sgl-kernel.yml` 的 Install 步骤,影响面为 sgl-kernel 的 CI 测试任务。

## 根因
正文未披露根因。由 diff 推断:该 CI 任务原本只安装 `torch==2.6.0` 与 pytest,未安装 torchvision;随后步骤会 `pip3 uninstall sgl-kernel -y` 并以 `--force-reinstall --no-deps` 重装 sgl-kernel wheel,环境中缺少 torchvision 时会报 `torchvision::nms not exist`(与标题一致),补装 torchvision 即可消除。

## 修复方案
`.github/workflows/pr-test-sgl-kernel.yml` 的 Install 步骤在安装 torch 时一并安装 torchvision:

```diff
-          pip3 install torch==2.6.0 && pip3 install pytest
+          pip3 install torch==2.6.0 torchvision && pip3 install pytest
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 CI 工作流的依赖安装步骤,不触碰运行时代码;为 sgl-kernel 的 CI 测试环境引入 torchvision 依赖。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
