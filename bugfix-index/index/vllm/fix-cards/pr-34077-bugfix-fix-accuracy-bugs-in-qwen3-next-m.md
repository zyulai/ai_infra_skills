# PR #34077: [BUGFIX] Fix accuracy bugs in Qwen3-Next MTP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34077](https://github.com/vllm-project/vllm/pull/34077) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-02-10 |
| labels | bug, qwen, ready, v1 |
| 热度 | 👍 0 · 💬 14 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Qwen3-Next MTP+CG 因 padded batch 处理不当有精度损失且被高并发掩盖;修正零长序列计数、state_indices 与拷贝形状。

## 问题现象
Qwen3-Next MTP+CG 存在精度损失,且该损失被掩盖:`lm_eval` 用高 `num_concurrent` 时精度正常,因为高并发下不使用 CG。验证:`vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 -tp 4 --no-enable-prefix-caching --speculative_config.method qwen3_next_mtp --speculative_config.num_speculative_tokens 3`,gsm8k 5-shot 于 `CONC={17,47,109}`:flexible-extract exact_match 0.8597 / 0.8529 / 0.8544,strict-match 0.8173 / 0.8127 / 0.8097,均与 baseline 对应。

## 根因
正文披露:问题全部出在 batch padded 部分的处理:
- 零长序列被计为 prefill;
- `state_indices` 没有考虑 padded 元素;
- 拷贝到持久缓冲时形状不匹配(shape mismatch)。

## 修复方案
`vllm/v1/attention/backends/gdn_attn.py` 的 `build()`:
- prefill 计数排除零长 padded 序列:`num_zero_len = (non_spec_query_lens == 0).sum().item()`,从非 spec 序列数中扣除;
- `spec_state_indices_tensor` 改为按 `spec_sequence_masks` 过滤 `block_table_tensor`,不再取全部行;
- `spec_query_start_loc` 截取 `query_start_loc[: num_spec_decodes + 1]`(padded 序列总在尾部,前缀累计值已正确);
- 持久 mask 拷贝改为 `spec_sequence_masks[:num_spec_decodes]`,修正形状不匹配;
- 新增强断言 `not (num_decodes > 0 and num_spec_decodes > 0)`(报 `num_decodes: ..., num_spec_decodes: ...`)。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:GDN attention metadata 构建在 spec decode + cudagraph 下的路径;padded/混合批次的 state_indices 与 mask 布局变化只影响按 diff 所及的 GDN backend 自身消费方;高并发(不启用 CG)路径行为与修复前一致。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
