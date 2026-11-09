# PR #5639: [Bugfix] Fix for inconsistent behaviour related to sampling and repetition penalties

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5639](https://github.com/vllm-project/vllm/pull/5639) |
| 作者 | tdoublep |
| 合并日期 | 2024-06-18 |
| labels | (none) |
| 热度 | 👍 5 · 💬 7 |
| 分类 | Sampler / Detokenization |
| bug 类型 | general |

## 一句话结论
do_penalties 在循环里即时求值,惩罚行为取决于 batch 内请求顺序;改为循环前预先一遍算出 do_penalties。

## 问题现象
采样与 repetition penalty 的行为不一致:正确与否取决于 batch 内请求的顺序。作者用两个并发请求即可复现——同一 prompt,第一个请求带 `temperature=0.0, max_tokens=5, repetition_penalty=2`,第二个带 `seed=99, max_tokens=4, top_k=1`。

## 根因
正文披露:`SamplingTensors` 中的 `prompt_tokens` 与 `output_tokens` 形状错误,根源在 `from_sampling_metadata` 里 `do_penalties` 是在循环中即时(on the fly)求值的:如果带 `repetition_penalty=2` 的请求先出现,则整个 batch 的 `do_penalties=True`,第二个请求的 `prompt_tokens`/`output_tokens` 也有长度 2(sampling_metadata.py L399-L400 处的拼接),行为正确;反过来,若不带 repetition_penalty 的请求先出现,这两个列表长度只有 1,导致采样行为错误。

## 修复方案
修改 `vllm/model_executor/sampling_metadata.py` 的 `from_sampling_metadata`:主循环里删除对 `prompt_tokens`/`output_tokens` 的内联条件拼接(prefill 分支的 `extend([] ...)` 与 `do_sample` 分支的逐 seq append),改为主循环之后用独立的 `if do_penalties:` 块对 `seq_groups` 再做一遍快速遍历统一收集,使 `do_penalties` 的判定提前完成、与请求顺序无关:

```diff
+        if do_penalties:
+            for seq_group in sampling_metadata.seq_groups:
+                seq_ids = seq_group.seq_ids
+                if (seq_group.is_prompt
+                        and sampling_params.prompt_logprobs is not None):
+                    prefill_len = len(seq_group.prompt_logprob_indices)
+                    prompt_tokens.extend([] for _ in range(prefill_len))
+                    output_tokens.extend([] for _ in range(prefill_len))
+                if seq_group.do_sample:
+                    for seq_id in seq_ids:
+                        seq_data = seq_group.seq_data[seq_id]
+                        prompt_tokens.append(seq_data.prompt_token_ids)
+                        output_tokens.append(seq_data.output_token_ids)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`SamplingTensors` 构建路径中 `prompt_tokens`/`output_tokens` 的组装顺序与时机;所有启用 penalty 的采样路径行为与旧正确顺序一致,未启用 penalty 的路径不进入新块(由 diff 推断)。prefill(prompt_logprobs)与 do_sample 两个分支的收集条件保持不变,仅执行时机后移。

## 关联
- Fixes #5607(正文首行 "Fixes #5607")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
