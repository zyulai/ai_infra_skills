# PR #52588: docs: fix incorrect --custom-skip-chat-template flag reference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52588](https://github.com/vllm-project/vllm/pull/52588) |
| 作者 | theamalsebastian |
| 合并日期 | 2026-08-17 |
| labels | documentation, ready, build-docs |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
基准测试文档引用了不存在的 `--custom-skip-chat-template` 标志,实际标志为 `--skip-chat-template`;更正文档中两处引用。

## 问题现象
`docs/benchmarking/cli.md` 中 custom 数据集的 `vllm bench serve` 示例命令与说明文字引用了一个不存在的标志 `--custom-skip-chat-template`。影响面:仅文档——按文档照抄命令的用户会因未知标志而报错(argparse 不识别该参数);运行时代码本身没有 bug。正文 Test Plan 说明通过 `grep -rn "skip-chat-template" vllm/ benchmarks/` 验证:该标志在真实 CLI 代码中定义为 `--skip-chat-template`(argparse `add_argument`、`action="store_true"`),代码里任何地方都没有 `custom-` 前缀版本。

## 根因
正文披露:文档把标志名写错了。标志在 `vllm/benchmarks/datasets/datasets.py` 中定义为 `--skip-chat-template`,而 `docs/benchmarking/cli.md` 引用成了 `--custom-skip-chat-template`(混淆了 `custom` 数据集名称与标志前缀)。正文未说明该笔误的引入来源。

## 修复方案
仅改 `docs/benchmarking/cli.md`,共两处:
- 示例命令中 `--custom-skip-chat-template` → `--skip-chat-template`:
  ```diff
  -  --custom-skip-chat-template \
  +  --skip-chat-template \
  ```
- 说明文字同步更正:「You can skip applying chat template if your data already has it by using `--skip-chat-template`。」
无任何代码改动;作者经 `git diff` 确认仅更新这两处引用。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围(对应 issue #52587)。
- 回归面:无功能性影响——纯文档改动,不触及任何代码路径;唯一的「行为变化」是按文档执行 `vllm bench serve --dataset-name custom` 示例的用户从此使用真实存在的 `--skip-chat-template` 标志,示例命令可正常运行。

## 关联
- [Fixes #52587](https://github.com/vllm-project/vllm/issues/52587)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
