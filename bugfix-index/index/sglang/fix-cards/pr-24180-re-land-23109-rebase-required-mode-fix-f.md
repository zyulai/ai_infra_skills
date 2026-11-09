# PR #24180: Re-land #23109: rebase-required mode + fix for grep-no-match abort

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24180](https://github.com/sgl-project/sglang/pull/24180) |
| 作者 | alisonshao |
| 合并日期 | 2026-05-08 |
| labels | documentation, Multi-modal, run-ci |
| 热度 | 👍 0 · 💬 13 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
check-maintenance 的 `MIN_BASE_SHA` 解析在 grep 无匹配时经 pipefail 中止脚本,阻塞全部 PR CI;`|| true` 兜底并重新落地 #23109 的 rebase-required 模式。

## 问题现象
#23109 曾被 #24179 回退。回退原因:当 issue body 没有 `MIN_BASE_SHA` 指令(正常情况)时,每个 job 的 `check-maintenance` 步骤在 0 秒内 exit 1,错误地阻塞所有 PR CI。本地可复现:

```bash
bash --noprofile --norc -eo pipefail -c '
  A=$(echo "" | grep "x" | head -n1)
  echo after
'
# -> exits 1, never prints "after"
```

## 根因
正文披露:`check-maintenance` action 的 `MIN_BASE_SHA` 解析器通过管道接 `grep`,而 `grep` 在无行匹配时 exit 1。`shell: bash` composite steps 默认以 `bash --noprofile --norc -eo pipefail {0}` 运行,exit 1 经 pipefail 传播,在任何后续检查运行之前中止整个脚本。

## 修复方案
- 修复 grep-no-match:把 `grep` 包进 brace group 并加 `|| true`,空匹配时干净地落到 `head`/`sed`/`awk`(正文 Fix 节;本地用空 body 与含指令两种输入验证)。
- 重新落地 #23109 的原始功能:为 `check-maintenance` 增加 rebase-required 模式。issue #21065 body 可包含 `MIN_BASE_SHA: <sha>` 指令;存在时(无论 issue 开或关),仅当 PR 分支已包含该 sha(GitHub compare API 状态 `ahead`/`identical`)才允许 CI,`behind`/`diverged` 则以 "Rebase Required" 阻塞;原 full-pause 维护门(issue OPEN 且无 SHA)行为不变;`bypass-maintenance` label 仍绕过两个门;API 错误 fail-open。
- diff 另有两处配套改动:`action.yml` 从 `gh` CLI 改为 curl + jq(self-hosted GPU runner 未装 gh,否则 rebase 门只在 ubuntu-latest 上生效);`scripts/ci/utils/slash_command_handler.py` 新增 `_check_rebase_gate` 预检,`/rerun-stage` 与 `/rerun-test` 在 dispatch 前短路并只发一条说明评论,避免每个 job 都在门上失败、浪费 runner 并产生 N 个 error annotation。

```diff
-        MIN_BASE_SHA=$(printf '%s' "$ISSUE_BODY" | tr -d '\r' | grep -iE '...' | head -n1 | sed -E '...' | awk '{print $1}')
+        MIN_BASE_SHA=$(
+          { printf '%s' "$ISSUE_BODY" | tr -d '\r' | grep -iE '...' || true; } \
+            | head -n1 | sed -E '...' | awk '{print $1}'
+        )
```

## 影响与回归风险
- 回归面:CI 基础设施路径——所有 PR 的 `check-maintenance` 步骤、`pr-test.yml` 门控行为(新增 rebase-required 门,行为矩阵见正文:issue closed 且无 SHA 时 CI 正常运行,即之前被误伤的场景)、`/rerun-stage`、`/rerun-test` 斜杠命令的 dispatch 前置检查;文档 `MAINTENANCE.md` 新增 Rebase-Required Mode 说明(只读第一行 `MIN_BASE_SHA:`,SHA 须为 7-40 位十六进制,畸形值忽略并在 job summary 警告)。

## 关联
- #23109
- #24179
- #21065

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
