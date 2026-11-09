# PR #54442: [Bugfix] Don't let a structured-output request sample from an unmasked row

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ArcheyChen |
| 合并 | 2026-10-01 |
| 分类 | Other (tier: major) |
| Label | bug, speculative-decoding, ready, mrv2, scheduler |
| Diff | 素材未提供 diffstat;正文注明 production diff 为 85 added / 3 removed lines(six files) |
| 链接 | https://github.com/vllm-project/vllm/pull/54442 |

## 一句话结论

fail-closed 修复:结构化输出 + spec decode 下 bitmask 行在 -1 padding 后全变 permissive,导致输出违反 grammar 的 token;现在把超限 draft 置为无效。

## 动机

`StructuredOutputManager.grammar_bitmask` 只在 `apply_bitmask` 成立时约束行,调度窗口里出现 `-1` placeholder 后 `seen_padding` 置位,其后所有行被填 `_full_mask`(全词表放行)。当 `-1` 表示 "validate_tokens 拒绝了该 draft" 时这是安全的(被拒 draft 自己的行仍被 mask,接受不到 permissive 行);但当 `-1` 表示 "draft 从未到达 scheduler" 时不安全——`DraftTokensHandler` 只有一个 slot 且每次 `execute_model` 覆写(`vllm/v1/worker/gpu/spec_decode/utils.py`),batch 中无结构化输出请求时还会整体清空 payload;而 worker 用 GPU 上自己的 `req_states.draft_tokens` 验证真实 draft,接受可以一路走进不约束任何东西的行,请求输出 grammar 不允许的 token(#54437 有完整链条、model-free 复现与现场测量)。

## 优化点

本 PR 是 #54437 的 fail-closed 半边:让 permissive 行无论 draft 因何丢失都不可达(draft 交接的身份修复是另一处改动,#56802,二者互补——其 lookup miss 路径恰好回退到本 PR 的失效逻辑)。机制:按结构化输出请求携带 bitmask 实际约束了几个前导 draft——scheduler `get_grammar_bitmask` 用 #48200 引入的 `strip_speculative_padding` 填 `GrammarOutput.num_acceptable_drafts`;worker 侧 `grammar_invalid_drafts` 据此在设备上构造 logit 行 mask(draft 的 local position 来自设备 `cu_num_logits` 派生的 `expanded_local_pos`,与 `apply_grammar_bitmask` 相同的 (request, position) 键控,自适应验证在设备上裁剪 draft 时仍正确);`sample()` 在应用 bitmask 后把 mask 交给 rejection sampler,验证用 `draft_sampled.masked_fill(mask, -1)`——`is_valid_draft = draft_sampled >= 0` 在接受到达 permissive 行之前钉死接受长度。无内核改动:`rejection_sampler_utils` 已对 `-1` 位置直接从目标分布重采样(不取 `max(p-q, 0)` 残差)。draft 全部 back-fill(正常情形)时返回 None、零开销;字段缺失(旧 scheduler/warmup)时保守失效整个窗口。`apply_sampling_params` 与 watermarking 仍拿真实 draft id,penalty/bad-words 不受影响。

## 关键改动

```text
vllm/v1/worker/gpu/structured_outputs.py: 新增 grammar_invalid_drafts(设备侧构造 logit 行 mask)
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py: 验证改用 verify_draft_sampled(masked_fill -1)
vllm/v1/core/sched/scheduler.py + v1/core/sched/output.py: get_grammar_bitmask 输出 num_acceptable_drafts
vllm/v1/worker/gpu/model_runner.py: sample() 计算 invalid_drafts 并传入 rejection sampler
vllm/v1/worker/gpu/sample/batch_shard.py: num_acceptable_drafts 按 bitmask 同样方式分片
```

关键摘录(invalid 判定核心):

```python
local_pos = input_batch.expanded_local_pos
row_limit = torch.repeat_interleave(
    async_tensor_h2d(limit, device=local_pos.device),
    cu_num_logits[1:] - cu_num_logits[:-1], ...)
return local_pos > row_limit
```

## 性能声明

无吞吐性能声明。模型评测(RTX 4090,`VLLM_USE_V2_MODEL_RUNNER=1`,Qwen3-8B + Qwen3-8B-speculator.eagle3,K=4,--async-scheduling,xgrammar):gsm8k limit 200 上 main 0.925 ± 0.0187 vs 本 PR 0.925 ± 0.0187(spec decode 未受影响的对照组)。生产数据:2 节点 x 2 GPU(pp=2, tp=2,DeepSeek-V4-Flash + dspark spec decode,K=7,xgrammar)部署中,修复前 `tool_choice=required` 与 `json_schema` 在有第二个结构化输出请求在驻时 100% 失败;修复后 72/72 并发 grammar 请求完成、零 `grammar rejected`、零引擎错误。另有 @xinli-sw 报告本 PR 修复 16x B200 上 pp=2 x tp=8 + DSpark(K=3)的部署。

## 验证与回归风险

新增 `tests/v1/worker/test_grammar_invalid_drafts.py`(计划 5 例:整窗、部分窗、无事可做、per-request 偏移、自适应验证不均匀重分配);未提交的端到端复现测试在 main 上 `emitted [7, 7, 99, 7]` 失败、本分支通过;Test Plan 全选 71 passed / 1 xfailed。影响路径:spec decode + structured outputs 交叉,`batch_shard` 分片路径,自适应验证;非 grammar 批次 `grammar_invalid_drafts` 返回 None、行为不变;`_verify` 签名新增 `verify_draft_sampled` 参数波及既有测试。明确非 #56802 的重复(该 PR 是身份修复半边)。

## 关联

Part of #54437;互补 PR #56802(draft 交接身份修复);复用 #48200 引入的 `strip_speculative_padding`;与 PR #52436、#52771 同属 spec decode + 结构化输出/缓存正确性区域。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
