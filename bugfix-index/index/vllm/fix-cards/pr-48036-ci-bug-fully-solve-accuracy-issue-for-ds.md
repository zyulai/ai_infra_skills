# PR #48036: [CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48036](https://github.com/vllm-project/vllm/pull/48036) |
| 作者 | yewentao256 |
| 合并日期 | 2026-07-14 |
| labels | bug, deepseek, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
DSv3.2+MTP+序列并行下 MTP proposer 拿到的 token 布局不完整导致精度问题,修复为 SP MoE 后 all-gather 恢复完整 token 行并放宽 SP 触发条件。

## 问题现象
DeepSeek-V3.2 + MTP + Sequence Parallel 组合部署下存在精度问题(标题所述;正文未展开具体错误现象)。正文以 lm_eval gsm8k 验证修复:开启 SP 时 `exact_match` flexible-extract 0.9606、strict-match 0.9598(±0.0054)。验证命令为 `vllm serve deepseek-ai/DeepSeek-V3.2 ... --enable-expert-parallel --tensor-parallel-size 8 --tokenizer-mode deepseek_v32 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` 加 `lm_eval --model local-completions ... --tasks gsm8k`。

## 根因
正文仅写「A through fix for #47902」,未直接披露根因细节。从 diff 描述改动所暗示的机制:MTP proposer 经 SP MoE 层后,`hidden_states`/`residual` 的 token 行数少于 `num_tokens`(SP 下 token 分散在 TP rank 上),而 MTP 路径直接消费该 hidden states 计算 logits 并循环回收 post-final-norm hidden,没有主模型那样的融合 all-reduce/恢复步骤,导致草稿模型在不完整 token 布局上运行,产生精度损失(由 diff 推断)。另外 `use_sequence_parallel_moe` 原先要求 `data_parallel_size > 1` 才启用,与实际 SP 适用条件不符(由 diff 推断)。

## 修复方案
- `vllm/config/parallel.py`:`use_sequence_parallel_moe` 删除 `and self.data_parallel_size > 1` 条件,SP MoE 不再强制要求 DP>1。
- `vllm/model_executor/models/deepseek_mtp.py`:新增 `_restore_full_token_layout_if_needed`——若 `hidden_states.shape[0] != num_tokens`,则 cat(hidden_states, residual) 后 `tensor_model_parallel_all_gather` 再按 `num_tokens` 截断并切分回两者;在 `SharedHead.forward` 的 mtp_block 之后调用。
- `vllm/models/deepseek_v32/nvidia/mtp.py`:forward 中 mtp_block 后同样调用该恢复函数,并把注释中 `reduce_results=False` 修正为 `skip_final_all_reduce`。关键摘录:

```python
def _restore_full_token_layout_if_needed(hidden_states, residual, num_tokens):
    """Restore full token rows for the MTP proposer after SP MoE layers."""
    if hidden_states.shape[0] == num_tokens:
        return hidden_states, residual
    combined_states = torch.cat([hidden_states, residual], dim=-1)
    combined_states = tensor_model_parallel_all_gather(combined_states, 0)
    combined_states = combined_states[:num_tokens]
    ...
```

## 影响与回归风险
- 回归面:使用 SP MoE 的 DeepSeek MTP 路径(`deepseek_mtp.py`、`deepseek_v32/nvidia/mtp.py`)——token 布局不完整时会多一次 all-gather 通信(开销);`use_sequence_parallel_moe` 条件放宽后,此前不启用 SP 的 DP=1、TP>1 + EP 配置现在会走 SP MoE,属行为变更;非 MTP、非 DeepSeek 路径不受影响。

## 关联
#47902(本 PR 为其彻底修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
