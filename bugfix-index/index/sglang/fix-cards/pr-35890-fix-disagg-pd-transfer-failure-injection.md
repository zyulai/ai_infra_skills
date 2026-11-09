# PR #35890: fix(disagg): PD transfer-failure injection was silently inert

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#35890](https://github.com/sgl-project/sglang/pull/35890) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-08-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PD 传输故障注入因 `EnvField.get()` 永不抛异常而完全失效,两个 CI 测试零注入空转通过;测试改用 `envs.override()` 并删除死代码。

## 问题现象
两个已注册 CI 测试(`test/registered/disaggregation/test_disaggregation_basic.py` 与 `test/registered/amd/disaggregation/test_disaggregation_basic.py`)设置 `DISAGGREGATION_TEST_FAILURE_PROB=0.05` 来演练 PD 传输失败恢复,但实际注入率始终为 0,二者的 retry-path 断言一直空洞通过(vacuously passing)。作者在 base commit 上复现:设 `DISAGGREGATION_TEST_FAILURE_PROB=0.05` 时 `_get_failure_prob()` 返回 `0.0`。

## 根因
正文披露:`_get_failure_prob()`(`python/sglang/srt/disaggregation/utils.py`)用 `try/except` 包裹 `envs.SGLANG_TEST_DISAGG_FAILURE_PROB.get()`,异常时回退读 legacy 名 `DISAGGREGATION_TEST_FAILURE_PROB`。但 `EnvField.get()` 从不抛异常——未设置时返回默认值 `0.0`,值不可解析时也仅告警并返回默认,因此 except 分支不可达,legacy 名被直接忽略。

## 修复方案
4 个文件,+13/-17,`environ.py` 不动:
- 两个故障注入测试改用 `envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(0.05)`,并在服务器启动前进入 override,让子进程继承该环境变量;
- 删除 legacy 名 `DISAGGREGATION_TEST_FAILURE_PROB`(test-only 且从未生效,不存在依赖它的可用行为),随之删除死 `try/except`;`_get_failure_prob` 变成一行包装后被内联进唯一调用者并删除;
- 移除 nixl 测试防御性清理中对应的 legacy `os.environ.pop`。

```python
def _poll_with_failure_injection(pollers) -> List[int]:
    if (failure_prob := envs.SGLANG_TEST_DISAGG_FAILURE_PROB.get()) > 0:
```

作者以改动后的 `_poll_with_failure_injection` 对 4000 个 poller 端到端验证:override(0.05) 时注入率 4.78%(191/4000),override 退出后注入停止;legacy 名仍无注入(与修复前一致)。两个 PD 测试本身因需多卡硬件未在本地运行。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动仅涉及 test-only 故障注入路径与两个测试;`DISAGGREGATION_TEST_FAILURE_PROB` 在仓库中不再出现。正文 NOTE:这两个测试将首次真正注入 5% 传输失败,若恢复路径在注入失效期间已退化,测试可能开始失败——那将暴露既有缺陷,而非本 PR 引入的回归。

## 关联
#35838, #35843, #35844, #35847, #35886

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
