# PR #9038: [Bugfix] Fix incorrect updates to num_computed_tokens in multi-step scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9038](https://github.com/vllm-project/vllm/pull/9038) |
| 作者 | varun-sundar-rabindranath |
| 合并日期 | 2024-10-06 |
| labels | ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
Multi-Step 调度下 `num_computed_tokens` 的更新不遵守语义(统计已算好 KV 槽位的 token 数);重构更新时机并新增单测。

## 问题现象
正文仅说明:`num_computed_tokens` 用于跟踪序列中已有 computed KV cache slot 的 token 数,该语义在 main 分支使用 Multi-Step 调度时不被遵守。正文未描述具体现象(如错误输出或崩溃的具体表现),bug 由 @LiuXiaoxuanPKU 发现。

## 根因
正文未披露根因(未说明具体哪条路径算错)。(由 diff 推断)旧逻辑把 prefill 与 decode 的更新分散在多处且相互冲突:multi-step + chunked-prefill 下 prefill 首步即转 decode,旧嵌套函数 `update_prefill_num_computed_tokens` 依赖 `num_outputs`/`is_first_step_output` 分支判断;`_advance_to_next_step` 对 multi-step prompt 只 `update_num_computed_tokens(1)`;speculative decoding 时还在 `process_outputs` 后做 `output_token_num - 1` 的补丁式修正——这些路径叠加导致 multi-step 下更新量与实际 computed token 数不一致。

## 修复方案
- `vllm/engine/llm_engine.py` 新增统一入口 `_update_num_computed_tokens_for_multi_step_prefill`:decode 直接返回(更新移到 token append 之后);multi-step + chunked-prefill 只在首步输出(或单次 burst)时按 `token_chunk_size` 更新;普通 multi-step 的 prompt 单步调度,总是更新。
- `_process_model_outputs` 删除旧的嵌套 `update_prefill_num_computed_tokens`,非 async 路径统一为 multi-step 走新函数、否则按 `token_chunk_size` 更新;删除 speculative 的 `-1` 修正及 `process_outputs` 的返回值约定(接口改为返回 None)。
- `_advance_to_next_step` 与 `multi_step.py` 的 `_process_seq_outputs`:在 `append_token_id` 时区分 prefill 采样 token(不更新)与 decode token(更新 +1)。
- `rocm_flash_attn.py` 的 `advance_step` 对 `turn_prefills_into_decodes` 加 assert(Multi-Step + Chunked-Prefill 专属参数,该后端不支持)。
- 新增 `tests/core/test_num_computed_tokens_update.py`,参数化 num_scheduler_steps(1/8)× chunked-prefill × eager,校验 prompt/decode/结束各阶段的计数。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围。
- 回归面:v1 引擎(LLMEngine)multi-step 调度的输出处理与 `_advance_to_next_step` 路径、multi-step output processor 的 `process_outputs` 签名(不再返回 token 数)、speculative decoding 下 num_computed_tokens 的计法、ROCm flash attention `advance_step` 对 chunked-prefill 组合显式报错。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
