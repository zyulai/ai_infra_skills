# PR #55229: [Bugfix][DP] Prevent mismatched collectives on duplicate resume

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55229](https://github.com/vllm-project/vllm/pull/55229) |
| 作者 | bvolpato |
| 合并日期 | 2026-11-11 |
| labels | bug, documentation, ready |
| 热度 | 👍 1 · 💬 8 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
重复 `resume_generation()` 可使部分 rank 进入 resume all-reduce 而收到请求的 rank 跳过,DP MoE 引擎挂起;改用 pause 共识标志 `ignore_start_dp_wave` 守卫。

## 问题现象
正文 Overview:当请求开启一个新 wave 时,一次重复的 `resume_generation()` 会挂起 data-parallel MoE 引擎——空闲 rank 进入 resume all-reduce,而收到请求的 rank 跳过它,双方在集合通信上不匹配。回归测试构造:先向空闲 rank 发 resume,再向 rank 0 提交请求并随后 resume rank 0。验证环境:B200 GPU、`ibm-research/PowerMoE-3b`,以 merge base 恢复 `vllm/v1/engine/core.py` 作基线,`RAY_ADDRESS=local DP_SIZE=2 TP_SIZE=1`/`TP_SIZE=2` 运行 `test_dp_duplicate_resume_while_one_rank_starts_a_wave`;基线下仅 rank 1 打出 `setting pause state to UNPAUSED`,随后 pytest `+++ Timeout +++`;修复后两种 TP 尺寸下请求都能完成。

## 根因
正文 Details 披露:`engines_running` 会在**单个** rank 获知有工作时立即变化,各 rank 得知新 wave 的时间不同,因此用它守卫 resume all-reduce 会让各 rank 得出不同决定、各自进出集合通信不对称。而 `ignore_start_dp_wave` 在 ranks 达成 pause 共识时、以及新 Elastic EP rank 加入时设置,用它守卫可保证每个 rank 同一决定。

## 修复方案
改 `vllm/v1/engine/core.py` 的 `resume_scheduler`,把守卫条件从 `engines_running` 换成 `ignore_start_dp_wave`:

```python
-        if self.engines_running:
+        # Set at pause consensus, so all ranks take the same branch.
+        if not self.ignore_start_dp_wave:
             logger.debug("Resume called while engines are not paused, ignoring.")
             return
```

同时更新 `docs/training/async_rl.md`(resume 前需等 `pause_generation()` 完成,DP MoE 引擎可拒绝过早的 resume;resume 后重复调用无效果),并新增上述 DP 回归测试。

## 影响与回归风险
- 影响窗口:正文未说明受影响版本/提交范围,不写。
- 回归面:按 diff,`resume_scheduler` 的早退分支条件变化,只影响 DP pause/resume 路径:未暂停的 rank 即使晚于其他 rank 获知 wave 也会跳过 resume collective;paused ranks 与新加入的 Elastic EP ranks 仍在 stepping 前同步;pause 尚在处理中的 resume 仍被拒绝。改动伴随文档更新与新增测试;作者指出 #59559 已把开 wave 的职责移到 engines 侧。

## 关联
- [#59559](https://github.com/vllm-project/vllm/pull/59559)
- [Review discussion](https://github.com/vllm-project/vllm/pull/55229#discussion_r4170893423)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
