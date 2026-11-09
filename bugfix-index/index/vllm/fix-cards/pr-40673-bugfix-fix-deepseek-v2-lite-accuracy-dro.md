# PR #40673: [Bugfix] Fix DeepSeek V2-Lite Accuracy drop

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40673](https://github.com/vllm-project/vllm/pull/40673) |
| 作者 | bnellnm |
| 合并日期 | 2026-04-23 |
| labels | bug, deepseek, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
#40560 引入 DeepSeek V2-Lite 精度回退(0.02→0.35);修复 MoE 共享专家输出的 all-reduce 路径,在开启 sequence parallel 时跳过早归约。

## 问题现象
DeepSeek V2-Lite 出现精度下降:修复前相关测试精度仅 0.02,修复后恢复到 0.35(正文 Test Result: "Accuracy went from 0.02 -> 0.35")。正文未给出更具体的日志或复现输出。

## 根因
正文说明该精度下降由 [#40560](https://github.com/vllm-project/vllm/pull/40560) 引入,未展开机制细节。由 diff 推断:问题出在 `moe_runner.py` 的 `_maybe_reduce_shared_expert_output` —— 当 combine kernel 已对 fused_output 做过归约时,共享专家输出需要单独 all-reduce;但该"early all-reduce"路径没有区分 sequence parallel(SP,即 TP=N、DP=M、EP)场景,SP 下模型侧有单独的 AG 步骤,再走 tensor_model_parallel_all_reduce 会对共享专家输出重复/错误归约,导致数值错误。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` 一处条件:

```diff
-        if shared_output is not None and self._fused_output_is_reduced:
+        if (
+            shared_output is not None
+            and not self.moe_config.is_sequence_parallel
+            and self._fused_output_is_reduced
+        ):
             shared_output = tensor_model_parallel_all_reduce(shared_output)
```

即 SP 开启时跳过对 shared_output 的早归约,交给模型侧已有的 AG 步骤。作者验证了 deepseek_v2_lite_ep_eplb 集成脚本、#39956 中的所有模型以及 CI MoE Refactor 测试。

## 影响与回归风险
- 影响窗口:#40560 合并之后、本 PR 之前的版本;正文以 PR 链接为界,未给出具体版本号。
- 回归面:fused MoE runner 的共享专家输出归约路径;所有走到 `_maybe_reduce_shared_expert_output` 且 `is_sequence_parallel` 为真的 MoE 配置行为改变(不再做该次 all-reduce),非 SP 配置不受影响。

## 关联
- https://github.com/vllm-project/vllm/pull/40560
- https://github.com/vllm-project/vllm/pull/39956

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
