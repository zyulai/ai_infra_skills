# PR #14237: [Bugfix] Fix DeepSeek MTP crash when using TP1ModelRunner with CUDA graph due to shape mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14237](https://github.com/vllm-project/vllm/pull/14237) |
| 作者 | pyc96 |
| 合并日期 | 2025-03-05 |
| labels | bug, ready, speculative-decoding |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
DeepSeek MTP + TP1 draft runner + CUDA graph 在 BS>1 时 hidden_states 长度不匹配导致断言崩溃,修复为剔除 padding 项。

## 问题现象
使用 DeepSeek MTP 模块、TP1 model runner 且开启 CUDA graph 时,服务端在 execute_model 处崩溃;仅在 batch size > 1 时出现。复现启动命令为 `python -m vllm.entrypoints.openai.api_server --disable-log-requests --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 32 --seed 0 --tensor-parallel-size 16 --model deepseek-ai/DeepSeek-R1 --trust-remote-code --num-speculative-tokens 3 --speculative-draft-tensor-parallel-size 1`。崩溃日志为 `Error executing method 'execute_model'. This might cause deadlock in distributed execution.`,最终在 `vllm/sequence.py` 的 HiddenStates `__post_init__` 断言 `assert len(self.seq_group_metadata_list) == len(self.hidden_states)` 处 AssertionError(调用链经 spec_decode_worker → smaller_tp_proposer_worker → multi_step_worker → top1_proposer)。修复后不再崩溃。

## 根因
正文披露:使用 TP1 model runner + CUDA graph 时 hidden_states 存在 shape mismatch,即 CUDA graph 运行路径下 hidden states 带有 padding 项,与 seq_group_metadata_list 长度不一致,触发 `HiddenStates.__post_init__` 的长度断言失败;正文说明修复方式是「removing padded items in hidden states」。从 diff 看,机制是:启用 CUDA graph 时 `hidden_states` 尺寸按 graph padding 的 batch 走,而 `sampling_metadata.selected_token_indices` 只对应真实请求数,因此按 `len(selected_token_indices)` 截断 hidden_states 前缀即可对齐(由 diff 推断)。

## 修复方案
改动集中在 `vllm/spec_decode/draft_model_runner.py` 的 `execute_model`:在 `return_hidden_states and is_fallback` 分支下,区分是否走 CUDA graph——若 `use_cuda_graph` 为真,则取 `model_input.sampling_metadata.selected_token_indices`,只保留 `hidden_states[:len(indices)]` 作为 `output.hidden_states`,剔除 padding;否则维持原样直接赋值。关键摘录:

```python
if self.return_hidden_states and is_fallback:
    if use_cuda_graph:
        indices = model_input.sampling_metadata\
          .selected_token_indices
        output.hidden_states = hidden_states[:len(indices)]
    else:
        output.hidden_states = hidden_states
```

(diff 中另含若干 docstring 行尾空白清理,无逻辑变化。)

## 影响与回归风险
- 回归面:仅影响 spec decode 路径下 `return_hidden_states` 且走 fallback/GPU multi-step 的 `draft_model_runner.execute_model` 分支;CUDA graph 场景下 hidden_states 输出从整段 padding 尺寸变为截断到真实 token 数,依赖完整 hidden_states 输出的下游(如 EAGLE 类 proposer 消费 hidden_states 的路径)需注意;非 CUDA graph 路径行为不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
