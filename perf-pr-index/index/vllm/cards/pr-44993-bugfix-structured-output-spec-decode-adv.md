# PR #44993: [Bugfix][Structured Output][Spec Decode] Advance grammar across reasoning boundary

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yuyue0225sc |
| 合并 | 2026-07-23 |
| 分类 | Speculative Decoding(tier: major) |
| Label | bug, structured-output, speculative-decoding, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/44993 |

## 一句话结论

修复异步调度+投机解码下 grammar 跨 reasoning 边界失效，Qwen3.5 MTP 场景 JSON 有效率 7/50→50/50。

## 动机

正文指出在 async scheduling + speculative decoding + Qwen 风格 reasoning parser 下，`vllm/v1/structured_output/__init__.py` 的 `should_advance` 对 JSON/regex/choice 约束会静默失效，两个 bug 相互叠加。Bug 1：由 `num_computed_tokens - num_output_placeholders` 算出的 delta window 在有 draft 被拒时（placeholder 计数保持 >0）会跳过 `</think>`，导致 `reasoning_ended` 永不翻转、grammar 从不生效，输出带 Markdown fence 或任意散文。Bug 2：即便 `reasoning_ended` 翻转，marker 步产生的 marker 之后内容 token 也从未进入 grammar FSM；下一步用初始态 grammar 准备 bitmask，模型再次吐出起始 token，`response_format=json_object` 返回 `{{...}`（HTTP 200 但客户端解析失败，即 #48228 症状）。PR 修复 #43388、#48228、#34650。

## 优化点

(1) `should_advance` 接受来自 scheduler 调用点的可选 `new_token_ids`，把它作为精确 delta window（修 Bug 1）；两个 draft-validation 调用点保留 placeholder fallback，行为不变。(2) 统一的边界路径覆盖所有约束类型：边界处 `should_advance` 记录 `reasoning_end_token_index` 并对每种约束返回 `True`，scheduler 随后用 `trim_reasoning_for_advance()` + `accept_tokens()` 只推进经验证的后缀（修 Bug 2）；`should_advance` 不再改动 grammar，只更新 reasoning 边界状态并决定是否让 scheduler 推进。(3) 删掉早先按约束类型分叉的保守 inline drain 与 STRUCTURAL_TAG 的 trim-and-advance 两条路径，理由是该位置已受 `grammar_bitmask` 约束且 rejection sampling 会阻止非法 draft 提交。(4) 一处有意语义变更：边界处 `accept_tokens` 被拒（正常流程不应发生）现在走 scheduler 标准错误路径终止请求，而非 warn-and-continue。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: update_from_output 传入 new_token_ids，边界处 trim + accept_tokens 推进后缀
vllm/v1/structured_output/__init__.py: should_advance 接受 new_token_ids、记录 reasoning_end_token_index、不再 mutate grammar
tests/v1/structured_output/test_reasoning_structured_output.py: 三个回归用例（Bug 1 精确 delta、无参 fallback、Bug 2 前缀 trim）
```

关键片段（scheduler 传入精确 delta）：

```diff
-            if new_token_ids and self.structured_output_manager.should_advance(request):
+            if new_token_ids and self.structured_output_manager.should_advance(
+                request, new_token_ids=new_token_ids
+            ):
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据（perf claims 检测为 none），未注明性能 benchmark 测试条件。正文的量化结果均为正确性/有效率验证（见「验证与回归风险」），例如 8×H20、`Qwen/Qwen3.5-35B-A3B`、`--reasoning-parser qwen3`、MTP k=4 下 `json_object` 由基线 7/50 提升到 50/50 干净 JSON。

## 验证与回归风险

8×H20、Qwen3.5-35B-A3B、qwen3 parser：`json_object` MTP k=4 50/50（main 基线 7/50，86% 失败）；strict tool calling / `STRUCTURAL_TAG` MTP k=1 50/50 valid `tool_calls`、0 HTTP 500（确认统一路径不回退 #44297）；单测 11 passed。独立验证：@estrella159 在 vLLM 0.24.0（Blackwell sm120）Qwen3.6-27B-FP8 strict-JSON 14.12/24→23.38/24（paired p=0.0078）、Qwen3.6-35B-A3B-FP8 12.75/24→20.25/24；@stefanwichmann GB10（DGX Spark）Qwen3.6-27B-FP8 10/10 有效 JSON、双花括号消失；@Mazyod gemma-4-E2B-it + MTP（k=2、L4、strict json_schema）在 conc 8/16/24/32 的 JSON corruption 由 75%/100%/96%/91% 降为 0%；@cresslank DeepSeek-V4-Flash-DSpark 由 7/20 → 20/20，另 120/120 liveness 通过。测试命令 `pytest tests/v1/structured_output/ -v`、`pytest tests/v1/spec_decode/test_mtp_structured_output.py -v`。风险：有意语义变更（边界处 accept_tokens 拒绝改为终止请求）；MoE 架构仍有残余差距，正文归因于 grammar 级 whitespace-liveness hazard 被 spec decode 放大。

## 关联

#44297（sibling intra-step bitmask leak，已合并，patch 站点独立）；#36138（已合并，单 token marker 的 spec-decode×structured-output bypass，placeholder 数学在 async+draft rejection 下下溢）；#43526（已关闭，同第一步）；#43424（open，多 token marker，代码路径互补）；修复的 issue：#43388、#48228、#34650。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
