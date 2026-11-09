# PR #60714: [Bugfix] Reject max_num_scheduled_tokens=0 at config time instead of hanging

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60714](https://github.com/vllm-project/vllm/pull/60714) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-10-09 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论

max_num_scheduled_tokens=0 通过校验后引擎静默挂死,把下界从 ge=0 收紧为 ge=1。

## 问题现象

用户传 `max_num_scheduled_tokens=0` 时配置校验通过,`LLM.generate()` 永久挂起且无任何报错。作者在 1×NVIDIA B200(Python 3.12, pydantic 2.13)上复现:engine 158 秒就绪后 `generate()` 约 362 秒无进展,直到 `timeout` 杀掉进程,`EXIT_CODE=124`,无异常无错误日志;nightly 镜像同样复现。修复后约 1 秒抛出 pydantic `ValidationError`(EXIT_CODE=1),报 "Input should be greater than or equal to 1"。

## 根因

正文 Details 披露完整根因:

- `SchedulerConfig.max_num_scheduled_tokens` 定义为 `Field(default=None, ge=0)`,所以 0 能通过校验;唯一的另一处 `<= 0` 检查(`VllmConfig._set_max_num_scheduled_tokens`)只在开投机解码时执行。`Scheduler.__init__` 因值 `is not None` 而原样保留,于是每次 `schedule()` 都从 `token_budget = 0` 开始;running/waiting 两个调度循环都只在 `while token_budget > 0` 下运行,永远不调度任何 token,请求滞留 waiting 且不抛任何异常。
- 来源:PR #44207 之前该字段无下界、调度器用真值判断,0 会静默回退到 `max_num_batched_tokens`;#44207 加了 `ge=0` 并把判断改成 `is not None`,但投机路径拒绝的是 `<= 0`,新下界只拒绝 `< 0`,这个 off-by-one 让 0 变成了真实的零预算。

## 修复方案

把 `vllm/config/scheduler.py` 中字段下界由 `ge=0` 收紧为 `ge=1`,0 在 config 构建时即被 pydantic 拒绝;并在 `tests/test_config.py` 新增回归测试(旧下界下报 "DID NOT RAISE ValidationError")。作者说明选择改配置校验而非调度器:调度器把零预算当作有意的"暂停"语义,行为本身正确;同族字段 `max_num_batched_tokens`、`max_num_seqs` 也已用 `ge=1`。已知限制:`@config` dataclass 不在赋值时校验,但当前仅内部代码事后赋值(值为 ge=1 的 `max_num_batched_tokens`),用户路径到不了调度器。

关键 diff 摘录:

```diff
-    max_num_scheduled_tokens: int | None = Field(default=None, ge=0)
+    max_num_scheduled_tokens: int | None = Field(default=None, ge=1)
```

## 影响与回归风险

- 影响窗口:自 #44207(基准 commit `51e0c579b`)引入 `ge=0` 起至本修复前。
- 回归面:仅一处字段校验边界。None(默认)、1 及更大值行为不变(含 ngram 投机解码下 None 仍解析为 2048);`SchedulerConfig`、`EngineArgs`、投机解码、CLI 四条用户路径传 0 都会从静默挂起变为启动即报 ValidationError;投机路径原有的 `<= 0` 检查保留未动。

## 关联

- [issue #60349](https://github.com/vllm-project/vllm/issues/60349)(本 PR 修复的 issue)
- [PR #44207](https://github.com/vllm-project/vllm/pull/44207)(引入该 bug 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
