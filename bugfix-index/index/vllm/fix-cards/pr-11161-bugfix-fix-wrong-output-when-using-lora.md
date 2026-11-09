# PR #11161: [BugFix] fix wrong output when using lora and num_scheduler_steps=8

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11161](https://github.com/vllm-project/vllm/pull/11161) |
| 作者 | sleepwalker2017 |
| 合并日期 | 2025-02-01 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
dummy run 加载的 dummy LoRA 未被及时移除而残留,导致 LoRA + num_scheduler_steps=8 输出错误;改为在 _dummy_run 内执行完前向后立即移除。

## 问题现象
标题指出使用 LoRA 且 `num_scheduler_steps=8` 时输出错误;正文仅写 "FIX issue ..." 并链接相关 issue(#9688、#11086、#12487),未展开具体复现步骤与日志,正文未描述更具体的现象。

## 根因
正文未披露根因。(由 diff 推断) `_dummy_run`(用于显存 profiling 的 dummy 前向)中存在一个提前返回分支:执行 `execute_model` 与 `torch.cuda.synchronize()` 后直接 `return`,dummy 期间加载的 LoRA 未被移除;而旧代码中统一的 `remove_all_loras` 清理只放在 `worker.determine_num_available_blocks` 末尾,覆盖不到这条路径,残留的 dummy LoRA 随后影响真实推理,产生错误输出。

## 修复方案
- `vllm/worker/model_runner.py` `_dummy_run`:在该提前返回分支中,`execute_model` + `synchronize` 之后新增:

```diff
+            if self.lora_config:
+                # Remove dummy loras.
+                assert self.lora_manager is not None
+                self.remove_all_loras()
             return
```

- `vllm/worker/worker.py` `determine_num_available_blocks`:删除末尾的 `if self.model_runner.lora_manager: self.model_runner.remove_all_loras()`(保留 `gc.collect()`),清理职责移入 `_dummy_run` 本身。

## 影响与回归风险
- 回归面:启用 `lora_config` 的所有部署——dummy run(显存 profiling 等路径)结束时机提前移除 LoRA;若存在依赖"determine_num_available_blocks 之后 LoRA 仍在"的隐式行为会受影响;multi-step scheduling(num_scheduler_steps>1)与 LoRA 组合是本次修复的目标场景;非 LoRA 部署不受影响(diff 变更极小,仅两处)。

## 关联
- https://github.com/vllm-project/vllm/issues/9688
- https://github.com/vllm-project/vllm/issues/11086
- https://github.com/vllm-project/vllm/pull/12487

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
