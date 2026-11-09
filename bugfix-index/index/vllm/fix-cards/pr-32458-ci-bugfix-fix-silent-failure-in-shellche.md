# PR #32458: [CI][BugFix] Fix silent failure in shellcheck hook and baseline exist…

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32458](https://github.com/vllm-project/vllm/pull/32458) |
| 作者 | junuxyz |
| 合并日期 | 2026-02-11 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
shellcheck hook 的 find 命令写法无效导致 CI 静默漏检;修正 find 并引入 ~350 条既有警告的 baseline,CI 只对新增警告失败。

## 问题现象
CI 没有可靠地运行 shellcheck:由于 find 调用写法无效,shellcheck hook 行为不正确,脚本问题可能悄然漏检。修正 find 命令后 CI 开始失败,暴露出约 350 个既有 shellcheck 问题(以轻微 lint 为主:作者统计 2 error / 303 info / 12 style / 34 warning),包括如 `tools/flashinfer-build.sh` 的 SC2269(变量自赋值)等。相关 issue:#32391。

## 根因
正文披露:CI 脚本中的 find 调用无效,导致 shellcheck 静默失败、问题溜过检查。旧命令为:
```bash
find . -name "*.sh" ".git" -prune -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0 | ...
```
其中 `".git"` 被当作位置参数传入,写法不合法;此外脚本只有 `set -e`,未开 `pipefail`/`-u`,失败也可能被吞掉(由 diff 推断:set -euo pipefail 的改动所暗示)。

## 修复方案
改 `tools/pre_commit/shellcheck.sh`:
1. `set -e` 改为 `set -euo pipefail`。
2. 把旧的无效 find 重写为 `collect` 函数,正确用法 `-path ./.git -prune -o -name "*.sh" -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0`,以 gcc 格式输出并解析成 `file:SCcode` 对、`sort -u` 去重。
3. 新增 baseline 机制:新增文件 `tools/pre_commit/shellcheck.baseline`(89 行,记录全部既有 `file:SCcode` 组合);用 `comm -23` 求当前警告与 baseline 的差集,只有不在 baseline 中的新警告才导致脚本以 exit 1 失败;报错时用 `shellcheck --include=<新 SC 码>` 只显示每个文件的新问题,避免 baseline 内问题刷屏。
4. 新增 `--generate-baseline` 标志用于重新生成 baseline;baseline 文件缺失时报错退出并提示生成命令。
所有改动限于 CI 与工具链,不修改运行时或示例行为。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 pre-commit shellcheck hook(`tools/pre_commit/shellcheck.sh`)与其 baseline 文件。影响:任何新脚本或脚本修改引入的新 shellcheck 警告都会使 CI 失败(这是修复目的);后续贡献者改动被 baseline 覆盖的脚本时,若新增了 baseline 中没有的警告码也会失败。作者验证:baseline 在位时脚本通过;删除 baseline 中一条条目后脚本正确报出 SC2269。后续计划:基线清理通过后续 PR 增量处理,清理完可移除 baseline 文件与相关逻辑(作者将另开 issue 跟踪)。

## 关联
- [#32391](https://github.com/vllm-project/vllm/issues/32391)(related issue)
- [全量问题日志 gist](https://gist.github.com/junuxyz/047a464666c6b5b3a618a78ec7971f35)
- [SC2269 基线条目验证提交](https://github.com/vllm-project/vllm/pull/32458/commits/5982a51afcaf4c4a3d8bf7b187cf02ce07fd036b#diff-5a4731c03224c7ead786176a289176fe8c3c7bef74158a608bf1d2844bab6fc9R79)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
