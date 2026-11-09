# PR #43091: [Model Runner V2][Spec Decode] Support spec decode with draft model

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wxsIcey |
| 合并 | 2026-09-28 |
| 分类 | Speculative Decoding(tier: major) |
| Label | speculative-decoding, ready, ci/build, v1, mrv2 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/43091 |

## 一句话结论

在 Model Runner V2 上支持独立 draft model 的 speculative decoding,平均接受长度与 V1 持平(2.80 vs 2.78)。

## 动机

MRV2 此前把 `draft_model` 列入不支持的 spec decode 特性列表(`_get_v2_model_runner_unsupported_features`),无法在 v2 执行路径上用独立小 draft LM 做 speculative decoding。

## 优化点

新增 `PlainDraftModelSpeculator`(继承 DraftModelSpeculator):与 EAGLE 不同,draft 模型完全独立于 target 模型运行。step 0 用 Triton 内核 `_prepare_prefill_inputs_kernel` 构造扩展 buffer——每请求输出 [accepted tokens | correction token | rejected slots](每请求额外 1 个 query slot,`num_extra_query_per_req=1`),rejected slot 写临时 KV、后续轮次被有效 token 覆盖;step 1..k-1 为单 token decode 步,seq_lens 在每次 forward 前自增,使 attention 读到上一步写入的 KV slot;draft 模型以独立 vllm_config 加载并打 `draft_model` tag。CUDA graph 尚不支持(显式 pass);`_dummy_run` 在扩展 draft batch 超出 token 预算时扣减,`_build_attn_metadata` 支持外部传入扩展 slot_mappings。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/draft_model/speculator.py(新): PlainDraftModelSpeculator + prefill 输入准备 Triton 内核
vllm/v1/worker/gpu/spec_decode/__init__.py: uses_draft_model() 分发到新 speculator
vllm/v1/worker/gpu/model_runner.py: _dummy_run 为每请求 correction slot 预留预算
vllm/config/vllm.py: 从 MRV2 不支持列表移除 draft_model
vllm/v1/worker/gpu/spec_decode/speculator.py: BaseSpeculator 增加 num_extra_query_per_req;slot_mappings 可外部传入
```

## 性能声明

Qwen3-32B + Qwen3-1.7B draft(num-spec-tokens 3),mt-bench 400 prompts、temp 0、output 256:MRV1 mean acceptance length 2.78(逐位 acceptance 0.75/0.57/0.45)、total_num_output_tokens 102355;MRV2 为 2.80(0.76/0.58/0.46)、102312。属一致性对比,非吞吐基准,PR 未注明吞吐数据。

## 验证与回归风险

验证经 examples/features/speculative_decoding/spec_decode_offline.py。风险:draft 独立前向使 prefill/decode 预算每请求多 1 slot(max_num_tokens 需容纳 max_num_tokens + max_num_reqs 的扩展 buffer);CUDA graph 未支持(禁用路径);attention metadata 构建的 step 偏移含 correction slot,与 EAGLE 语义不同;扩展 buffer 与 expanded_slot_mappings 为常驻显存分配。

## 关联

同分类 Speculative Decoding:#38174(draft_model + 异构词表)、#50272(MRV2 spec decode bugfix)、#20401、#45635。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
