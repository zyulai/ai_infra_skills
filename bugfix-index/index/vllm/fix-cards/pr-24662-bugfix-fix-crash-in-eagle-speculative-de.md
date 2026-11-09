# PR #24662: [BUGFIX] Fix crash in Eagle Speculative Decoding models when exceeding draft model length

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24662](https://github.com/vllm-project/vllm/pull/24662) |
| 作者 | AlonKejzman |
| 合并日期 | 2025-09-25 |
| labels | documentation, structured-output, frontend, speculative-decoding, ready, v1, multi-modality, qwen |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
输入超过 draft 模型长度时 EAGLE 投机解码崩溃;增加 input_fits_in_drafter 检查,超长输入不再走 drafter 提案。

## 问题现象
PR 标题披露:当输入长度超过 draft 模型(drafter)的 max_model_len 时,Eagle 投机解码模型会崩溃。正文的 Purpose 描述了目标场景:输入可能超过 drafter 模型长度、但不超过 verifier(主模型)长度的环境下运行 Eagle 投机解码。具体报错信息与堆栈正文未提供。触发条件为输入序列长度落在 drafter 与 verifier 的最大长度之间。

## 根因
正文未披露根因。由 diff 推断:原逻辑在启用 EAGLE 且未关闭 padded drafter batch 时,直接把 GPU 采样出的 token 喂给 `propose_draft_token_ids`,完全没有检查当前输入序列长度是否超出 draft 模型自身的 `max_model_len`(draft 模型的长度可以配置得比 verifier 主模型短),超长输入进入 drafter 后越界,导致崩溃。

## 修复方案
改动 `vllm/v1/worker/gpu_model_runner.py` 中调用 `propose_draft_token_ids` 的分支:

1. 计算 drafter 的有效最大长度:优先取 `speculative_config.draft_model_config.max_model_len`,否则回退到 `self.max_model_len` / `model_config.max_model_len`;
2. 计算 `input_fits_in_drafter` 并同时作为两条提案路径(padded batch EAGLE 路径与 ngram 等 CPU 路径)的执行条件。关键 diff 摘录:

```python
input_fits_in_drafter = spec_decode_common_attn_metadata and (
    spec_decode_common_attn_metadata.seq_lens.max() +
    self.speculative_config.num_speculative_tokens
    <= effective_drafter_max_model_len)
if use_padded_batch_for_eagle and input_fits_in_drafter:
```

以及第二处:`if (self.speculative_config and not use_padded_batch_for_eagle and input_fits_in_drafter):`

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:两条投机解码提案路径(EAGLE padded batch 路径与 ngram 等基于 CPU 采样 token 的路径)都新增了同一长度门槛;输入超出 drafter 长度时会跳过本步 draft 提案,即该步投机加速失效(退化为正常解码)但不崩溃。`input_fits_in_drafter` 依赖 `spec_decode_common_attn_metadata` 非空的短路判断,该变量为 None 时不走提案。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
