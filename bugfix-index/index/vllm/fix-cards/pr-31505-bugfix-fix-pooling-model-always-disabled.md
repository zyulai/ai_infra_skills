# PR #31505: [Bugfix]Fix pooling model always disabled due to incorrect PP rank check

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31505](https://github.com/vllm-project/vllm/pull/31505) |
| 作者 | vintipandey |
| 合并日期 | 2025-12-30 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`broadcast_pp_output` 的判断条件 `len(get_pp_group().ranks) > 0` 恒为真,使 PP=1 时 pooling 模型走错分支触发 `assert not self.is_pooling_model` 而被禁用;条件改为 `> 1` 修复。

## 问题现象
pooling 模型(带 classification head 等)推理必然失败:使用 llama3 8b 分类头模型手动验证,修复前推理直接因断言 `assert not self.is_pooling_model` 报错,修复后 batch 推理正常运行。测试配置为 `tensor_parallel_size = 1, pipeline_parallel_size = 1`。即单 PP rank 的常规部署(external_launcher 后端)下所有 pooling 模型都被禁用。

## 根因
PR 正文披露:#24810 对 `gpu_model_runner.py` 的修改引入此 bug——`broadcast_pp_output` 的赋值逻辑为 `distributed_executor_backend == "external_launcher" and len(get_pp_group().ranks) > 0`,而 `get_pp_group()` 至少有 1 个 rank(在 `vllm/distributed/parallel_state.py` 中设置),所以该条件恒为 true。一旦为 true,后续分支进入 `else: # if self.broadcast_pp_output` 里的 "Rare case" 路径,执行 `assert not self.is_pooling_model`,pooling 模型于是必然断言失败。#24810 之前 pooling 分支的判定基于 `get_pp_group().is_last_rank`,单 rank 时同样满足,会正常走 `self._pool(...)` 返回。

## 修复方案
`vllm/v1/worker/gpu_model_runner.py` 中把 `broadcast_pp_output` 的 rank 数判断从 `> 0` 改为 `> 1`:

```diff
         self.broadcast_pp_output = (
             self.parallel_config.distributed_executor_backend == "external_launcher"
-            and len(get_pp_group().ranks) > 0
+            and len(get_pp_group().ranks) > 1
         )
```

即只有 PP ranks 多于 1 个才置位广播标志;`ranks == 1` 时 `pipeline_parallel_size` 为 1,本就不需要广播,pooling 模型恢复走正常的 pooling 分支。

## 影响与回归风险
- 影响窗口:#24810 合并之后、本 PR 之前;具体版本号正文未给出。
- 回归面:仅改动 `gpu_model_runner.py` 中 `broadcast_pp_output` 的初始化一行,影响使用 `external_launcher` 执行后端时的输出广播路径。多 PP rank(ranks > 1)场景行为不变;单 PP rank 场景从「恒广播并断言禁用 pooling」恢复为「不广播、pooling 正常执行」。对非 external_launcher 后端无影响(条件本就不满足)。

## 关联
- #24810(引入 bug 的 PR): https://github.com/vllm-project/vllm/pull/24810

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
