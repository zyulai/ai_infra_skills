# PR #10467: [Misc] Add multipstep chunked-prefill support for FlashInfer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elfiegg |
| 合并 | 2025-01-15 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/10467 |

## 一句话结论

让 FlashInfer 后端支持 multi-step + chunked-prefill 组合调度:第一步后 prefill token 转为 decode token 继续多步执行。

## 动机

vLLM 的 multi-step scheduling(chunked-prefill 场景,`num_scheduler_steps > 1`)此前只支持 `FLASH_ATTN` 等后端,FlashInfer 在 `advance_step` 中直接断言拒绝 `turn_prefills_into_decodes`("Chunked prefill is not supported with flashinfer yet")。要启用多步调度以减少调度器/CPU 开销的用户必须换注意力后端或放弃 chunked-prefill 组合。

## 优化点

机制上打通 FlashInfer 的多步 chunked-prefill 路径:multi-step 第一步中所有 prefill 转为 decode,`advance_step` 据此在原地更新 metadata(`num_decode_tokens += num_prefills`、清零 prefill 计数、`max_query_len = 1`、截断 `slot_mapping`);CUDA graph 场景下,`begin_forward` 仅在纯 decode batch(`num_prefills == 0`)时走 graph,避免 prefill 请求被错误地塞进 graph;`advance_step_flashinfer_kernel` 新增 padding 处理——`blockIdx.x == 0` 时把 graph padding 行的 `input_tokens/input_positions` 置 0、`slot_mapping` 置 -1。`MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS` 由仅 `FLASH_ATTN` 扩为 `[FLASH_ATTN, FLASHINFER]`。附带把 `profile_run` 包进 `set_in_profile_run` context manager(设置 `in_profile_run` 标志)。

## 关键改动

```text
vllm/attention/backends/flashinfer.py: advance_step 支持 turn_prefills_into_decodes;begin_forward 区分 decode
csrc/prepare_inputs/advance_step.cu: CUDA graph padding 行的 slot/token 清理
vllm/worker/multi_step_model_runner.py: 后端白名单加入 FLASHINFER
tests/multi_step/test_correctness_llm.py: 参数化 attention_backend=[FLASH_ATTN, FLASHINFER]、enforce_eager 两态
```

## 性能声明

PR 未提供 benchmark 数据,未注明测试条件;正文仅描述功能支持(multi-step scheduling for chunked-prefill on FlashInfer)。multi-step 调度本身的收益(摊薄调度开销)在 PR 中未量化。

## 验证与回归风险

扩展 `tests/multi_step/test_correctness_llm.py`:multi-step LLM 正确性测试新增 `FLASHINFER` 后端参数化与 `enforce_eager=[True, False]`(即覆盖 CUDA graph 路径),prompt-logprobs 与 chunked-prefill+APC 测试固定走 FLASH_ATTN。风险面:`advance_step.cu` 的 padding 逻辑影响所有 FlashInfer 多步 decode;`begin_forward` 的 graph 判定变化只应影响 prefill 与 decode 混合调度的 multi-step 场景;PR 正文注明 FlashInfer 尚不支持 speculative decoding + chunked-prefill + multi-step 组合(代码中以 assert 保留)。

## 关联

同区域:multi-step chunked-prefill 对 `FLASH_ATTN` 的既有支持(`MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS` 原有成员);FlashInfer attention 后端相关 PR 同属 Attention/Kernel 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
