# PR #18910: Fix flaky Qwen3-Next KL divergence tests by reverting mamba slot release

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18910](https://github.com/sgl-project/sglang/pull/18910) |
| 作者 | alisonshao |
| 合并日期 | 2026-02-18 |
| labels | (none) |
| 热度 | 👍 1 · 💬 21 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
调度失败时释放 COW 分配的 mamba 槽位导致重调度后状态从 radix cache 重建,引入数值差异使 Qwen3-Next KL 测试 flaky;回退该释放逻辑。

## 问题现象
Qwen3-Next KL divergence 测试变得非确定、大量失败:被测文件 `test_qwen3_next_models_mtp.py`、`test_qwen3_next_models.py`。CI 数据显示 slot release 改动上线前通过率约 90%(2 月 7-13 日 20 次 H100 运行 18 次过),之后跌到约 35%(2 月 13-18 日 20 次仅 7 次过);同一 commit `fd5a45d5c` 在 2 月 15 日既产出 PASS 又产出 FAIL,确认测试已非确定。失败表现为 outlier KL divergence 值(0.12-0.27)拉高算术均值超过测试阈值。

## 根因
PR 正文披露:当调度在 `init_next_round_input` 已完成 COW 分配之后失败,释放 mamba pool 槽位会导致重调度时状态从 radix cache 重建(取最后一个 tracking point 的状态),对部分样本引入轻微数值差异,表现为 outlier KL 值。CI 数据证实相关性:slot release 改动前后通过率从 ~90% 跌至 ~35%,同 commit 同日 PASS/FAIL 并存,证明非确定性来自该改动而非环境抖动。

## 修复方案
回退 `python/sglang/srt/managers/scheduler.py` 中调度失败路径上的 mamba 槽位释放逻辑:
1. 删除 `_get_new_batch_prefill_raw` 中 `res != AddReqResult.CONTINUE` 分支里的 `self.maybe_release_mamba_cache(req)` 调用。
2. 整体删除 `maybe_release_mamba_cache` 方法(其原逻辑为:非 disaggregation DECODE 模式下,若 `req.mamba_pool_idx` 非空则 free 并置 None,以避免 `check_memory()` 误报「内存泄漏」崩溃服务器)。

关键 diff(删除的调用点):

```diff
             if res != AddReqResult.CONTINUE:
-                self.maybe_release_mamba_cache(req)
-
                 if res == AddReqResult.NO_TOKEN:
```

Test plan:验证 Qwen3-Next KL 测试通过;验证正常负载下无 mamba slot 泄漏回归。

## 影响与回归风险
- 影响窗口:正文未给出具体版本,但指出回归由引入 `maybe_release_mamba_cache` 的 slot release 改动触发(2 月 13-18 日 CI 数据窗口)。
- 回归面:调度失败后 COW 分配的 mamba 槽位不再被主动释放,槽位由下一次调度轮经 `match_prefix()` 安全重新分配,代价是等待请求持有槽位期间占用内存;被删除方法 docstring 中提到的两个伴生问题——`check_memory()` 可能误报泄漏、以及 disaggregation DECODE 模式下 `self_check_during_idle` 跳过内存检查的 workaround——随方法一并移除,需关注这两个路径的后续处理。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
