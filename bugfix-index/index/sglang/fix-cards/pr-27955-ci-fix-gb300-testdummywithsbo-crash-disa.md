# PR #27955: [CI] Fix GB300 TestDummyWithSBO crash: disable NaN assert and coredump for dummy weights

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27955](https://github.com/sgl-project/sglang/pull/27955) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-11 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
GB300 上 TestDummyWithSBO 因 dummy 权重合法产生 NaN logits 触发 CI async assert 杀死 scheduler;对该测试关闭 assert 与 coredump 机制。

## 问题现象
GB300 上 TestDummyWithSBO 崩溃:dummy random weights 会合法地产生 NaN logits,触发 #27883 加入的 CI async NaN assert,在 warmup 阶段杀死 scheduler。正文未附具体报错日志。

## 根因
正文披露:dummy random weights 产生 NaN logits 是预期(合法)行为,但 #27883 引入的 async NaN assert 及配套 crash 机制(async assert、GPU 异常 coredump、crash 时 coredump)会把它当作致命错误处理,导致 scheduler 在 warmup 期间被杀。修复思路是对该测试关闭这套 crash 机制,让 NaN 被 sanitize 并输出节流 warning,而不是中断运行。

## 修复方案
`test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py` 的 `setUpClass` 子进程环境变量中新增四个关闭项:

```diff
+                "SGLANG_ENABLE_ASYNC_ASSERT": "0",
+                "SGLANG_CUDA_COREDUMP": "0",
+                # Already injected into os.environ by the test process when
+                # SGLANG_CUDA_COREDUMP=1, so it must be overridden explicitly.
+                "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0",
+                "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
```

其中 `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` 需要显式覆盖,因为测试进程在 `SGLANG_CUDA_COREDUMP=1` 时已把它注入 `os.environ`(diff 注释)。

## 影响与回归风险
- 回归面:仅 `test_deepseek_v3_cutedsl_4gpu.py` 这一个测试的子进程环境;被测模型路径与 CI 其他测试不受影响(由 diff 推断)。该测试下 NaN 不再触发崩溃,而是被 sanitize 并伴随节流 warning(正文)。

## 关联
- #27883

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
