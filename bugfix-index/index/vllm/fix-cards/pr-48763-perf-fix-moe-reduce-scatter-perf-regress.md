# PR #48763: [Perf] Fix moe `reduce_scatter` perf regression by removing additional comm, 5% E2E throughput gain back.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48763](https://github.com/vllm-project/vllm/pull/48763) |
| 作者 | yewentao256 |
| 合并日期 | 2026-07-25 |
| labels | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, needs-rebase, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, rust |
| 热度 | 👍 3 · 💬 4 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
MTP proposer 在 SP MoE 后用 cat+all-gather(2H)+all-reduce 恢复布局,多一次集合通信;改为 final norm 后单次 all-gather(H)。

## 问题现象
MoE reduce_scatter 路径的端到端性能回归:GLM-5.2-FP8 TP4 + EP + MTP(random 8192-in/1-out,128 prompts)下,main 总 token 吞吐 324671.20/346261.13 tok/s、请求吞吐 39.63/42.26 req/s;修复后 381839.84 tok/s、46.61 req/s(标题称找回 5% E2E 吞吐)。GSM8K 精度校验 flexible-extract 0.9439 / strict-match 0.9447,无精度回退。

## 根因
正文披露:SP MoE 层之后,MTP proposer 经 `_restore_full_token_layout_if_needed` 恢复完整 token 布局的做法是先 `cat(hidden_states, residual)` 再 all-gather 后 split,叠加 hidden 的 all-reduce——即 `all-gather(2H) + all-reduce(H)` 两次集合通信;而 final RMSNorm 用本地 residual 在各 rank 上算完后,只需一次 `all-gather(H)`(TP hidden states 按 P 倍复制后截断)即可得到同样结果。该回归来自 #48036 的方案,#48657 是另一备选。

## 修复方案
删除 `_restore_full_token_layout_if_needed`,把 all-gather 挪到 final norm 之后:
- `vllm/model_executor/models/deepseek_mtp.py`:`hidden_states = residual + hidden_states` 后,若 `use_sequence_parallel_moe` 则 `tensor_model_parallel_all_gather(hidden_states, 0)` 并截到 `positions.shape[0]`;
- `vllm/models/deepseek_v32/nvidia/mtp.py`:非 SP 仍走 all-reduce;SP 时先 `shared_head.norm(hidden_states, residual)` 再 all-gather + 截断:

```python
hidden_states, _ = self.shared_head.norm(hidden_states, residual)
if is_sequence_parallel:
    hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
    hidden_states = hidden_states[: positions.shape[0]]
```

## 影响与回归风险
- 影响窗口:回归由 #48036 引入(正文未写版本号)。
- 回归面:两条 MTP proposer 前向路径(deepseek_mtp.py、deepseek_v32/nvidia/mtp.py)的通信顺序改变——SP 模式从"norm 前 gather 拼接态"变为"norm 后 gather";非 SP 路径仍走 all-reduce,不变。验证仅覆盖 GLM-5.2-FP8 TP4+MTP 的吞吐与 GSM8K,其他模型/拓扑未在正文列出。

## 关联
- [#46654](https://github.com/vllm-project/vllm/issues/46654)(所属工作)
- [#48036](https://github.com/vllm-project/vllm/pull/48036)(引入回归的 PR)
- [#48657](https://github.com/vllm-project/vllm/pull/48657)(替代方案)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
