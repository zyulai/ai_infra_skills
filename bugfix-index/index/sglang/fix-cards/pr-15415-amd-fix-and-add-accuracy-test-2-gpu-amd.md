# PR #15415: [AMD] Fix and add accuracy-test-2-gpu-amd back

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15415](https://github.com/sgl-project/sglang/pull/15415) |
| 作者 | yctseng0211 |
| 合并日期 | 2025-12-19 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
2-GPU AMD 精度测试此前被禁用;原因是 aiter custom_allreduce 拉低分数(Humaneval 0.268→0.42x),测试命令加 SGLANG_USE_AITER_AR=0 关闭之并恢复该 CI job。

## 问题现象
`accuracy-test-2-gpu-amd` CI job(运行 `test_moe_eval_accuracy_large.py`)此前被临时禁用:job 的 if 条件里写死 `false &&`(workflow 注释注明跟踪于 issue #13107),且不在 `pr-test-amd-finish` 的 needs 里。禁用前的测试分数为 0.268(见正文 Before 数据)。

## 根因
正文给出的机制说明:该测试精度差是因为 aiter 的 custom_allreduce——在测试命令中加入 `SGLANG_USE_AITER_AR=0` 关闭它后,分数即恢复(MI-325 上 Humaneval pass@1 由 0.268 升至 0.422/0.421)。即 2-GPU 测试下 aiter custom_allreduce 的结果正确性问题导致测试长期不过而被禁用。

## 修复方案
改 `.github/workflows/pr-test-amd.yml` 三处:

1. 删除 `accuracy-test-2-gpu-amd` job if 条件中的 `false &&`,重新启用;
2. 评测命令增加 `-e SGLANG_USE_AITER_AR=0`,在该测试中禁用 aiter custom_allreduce:

```diff
-          bash scripts/ci/amd_ci_exec.sh -e SGLANG_USE_AITER=0 -e HF_HUB_ENABLE_HF_TRANSFER=0 python3 test_moe_eval_accuracy_large.py
+          bash scripts/ci/amd_ci_exec.sh -e SGLANG_USE_AITER_AR=0 -e SGLANG_USE_AITER=0 -e HF_HUB_ENABLE_HF_TRANSFER=0 python3 test_moe_eval_accuracy_large.py
```

3. `pr-test-amd-finish` 的 needs 列表重新加入 `accuracy-test-2-gpu-amd`。

验证(MI-325, Mixtral-8x7B-Instruct-v0.1):`SGLANG_USE_AITER_AR=0, num_threads=1024` 时 Humaneval 0.268→0.422(406.697s,MMLU 0.633);`SGLANG_USE_AITER_AR=1(default), num_threads=1` 时 0.268→0.421(1096.271s,MMLU 0.640),3 个测试均 OK。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 CI workflow(`.github/workflows/pr-test-amd.yml`),不触及任何运行时代码。影响是 2-GPU AMD 精度测试重新在 CI 中跑(带 `SGLANG_USE_AITER_AR=0`),CI 时长增加(该组测试约 407s~1096s);生产/用户路径零改动,aiter custom_allreduce 本身的正确性问题并未在此 PR 中修复,只是测试侧绕开。

## 关联
- 正文未列出(diff 中 workflow 注释提到的跟踪 issue 为 #13107)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
