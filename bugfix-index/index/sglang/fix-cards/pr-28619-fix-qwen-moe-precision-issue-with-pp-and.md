# PR #28619: Fix Qwen MoE precision issue with PP and all-reduce fusion

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28619](https://github.com/sgl-project/sglang/pull/28619) |
| 作者 | 1e4ves |
| 合并日期 | 2026-06-22 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
PP 边界处 `_sglang_needs_allreduce_fusion` 标记不随 PPProxyTensor 传递,待处理的 post-expert all-reduce 被整段跳过导致 KL 漂移;边界处先 flush 修复。

## 问题现象
Qwen MoE 在 pipeline parallelism 与 tensor parallelism 及 FlashInfer all-reduce fusion 同时启用时出现精度问题,表现为 PP+TP 运行中更大的 KL drift(正文 Fixes #27744)。

## 根因
正文披露:开启 all-reduce fusion 后,post-expert all-reduce 可以被延迟并融合进下一层的 `prepare_attn` 路径,由 Python 侧标记 `_sglang_needs_allreduce_fusion` 跟踪,正常情况下由下一层消费。但在 PP 边界,hidden states 被包成 `PPProxyTensors` 发往下一个 pipeline stage,该 Python 侧融合标记不会跨边界保留,于是一次待处理的 post-expert all-reduce 被完全跳过,造成 PP+TP 运行更大的 KL 漂移。正文并指出 EP 也有同样问题(EP 同样需要 post-expert all-reduce)。

## 修复方案
`python/sglang/srt/models/qwen2_moe.py` 的 `forward`:在返回 `PPProxyTensors` 前先冲刷(flush)待处理的 MoE/EP all-reduce,确保延迟的规约在张量交给下一个 PP stage 前完成:
```python
if (hidden_states is not None
        and hasattr(hidden_states, "_sglang_needs_allreduce_fusion")
        and hidden_states._sglang_needs_allreduce_fusion):
    if get_moe_expert_parallel_world_size() > 1:
        hidden_states = moe_expert_parallel_all_reduce(hidden_states)
    if get_moe_tensor_parallel_world_size() > 1:
        hidden_states = moe_tensor_model_parallel_all_reduce(hidden_states)
    hidden_states._sglang_needs_allreduce_fusion = False
```
按 EP 世界大小与 TP 世界大小分别执行 `moe_expert_parallel_all_reduce` / `moe_tensor_model_parallel_all_reduce` 后清掉标记。验证:正文附准确性测试截图(checklist 对应项未勾选)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `qwen2_moe.py` 非最后一个 PP rank 的返回路径。仅当 hidden states 携带融合标记时才触发额外 all-reduce,未启用 all-reduce fusion / 非 PP 的路径不变;PP 边界处会多一次同步规约,理论上有轻微延迟开销(正文未提供速度测试)。

## 关联
- [#27744](https://github.com/sgl-project/sglang/issues/27744)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
