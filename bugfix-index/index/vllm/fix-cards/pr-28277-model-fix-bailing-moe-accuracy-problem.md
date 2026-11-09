# PR #28277: [Model] Fix bailing_moe accuracy problem

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28277](https://github.com/vllm-project/vllm/pull/28277) |
| 作者 | zhaozx-cn |
| 合并日期 | 2025-11-14 |
| labels | ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
vllm-ascend 的 all2all MoE 策略下,SharedFusedMoE 后无需再做 all-reduce;bailing_moe 改用 `maybe_all_reduce_tensor_model_parallel` 修复精度问题。

## 问题现象
在 vllm-ascend 上使用基于 all2all 的 MoE 策略跑 bailing_moe 时出现精度(accuracy)问题。正文未给出具体复现命令、精度数值或日志,也未描述影响版本范围。

## 根因
PR 正文披露:当使用 all2all-based MoE 策略(vllm-ascend)时,SharedFusedMoE forward 之后不需要再执行 all-reduce;而 bailing_moe 在 `tp_size > 1` 时无条件调用 `tensor_model_parallel_all_reduce`,导致精度问题。正文并指出 glm4_moe、deepseek_v2 等其他 MoE 模型此前已通过改用 `maybe_all_reduce_tensor_model_parallel` 修复过同类问题,本 PR 将同样的做法套用到 bailing_moe。

## 修复方案
改 `vllm/model_executor/models/bailing_moe.py` 的 decoder 层 forward:在 `tp_size > 1` 分支中,把直接调用 `tensor_model_parallel_all_reduce(final_hidden_states)` 换成 `self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)`,由 SharedFusedMoE 的 experts 模块按当前 MoE 通信策略判断是否需要 all-reduce;同时删除不再使用的 `tensor_model_parallel_all_reduce` import。

```diff
         if self.tp_size > 1:
-            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
+            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
+                final_hidden_states
+            )
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅影响 bailing_moe 在 TP>1 下的 forward 输出路径。all-reduce 决策改为由 experts 的 `maybe_all_reduce_tensor_model_parallel` 按后端策略判断;按正文,glm4_moe 与 deepseek_v2 已采用同一模式,风险与这两个模型一致。非 all2all 的普通 TP 场景预期仍会执行 all-reduce(由该 maybe_ 方法保证),但这一行为由 diff 与既有模型类比推断(由 diff 推断)。

## 关联
正文未列出具体 issue/PR 链接(仅文字提及 glm4_moe、deepseek_v2 曾做同样修复)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
