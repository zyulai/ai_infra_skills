# PR #21030: Fix streaming logprobs corruption caused by shared mutable list reference

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21030](https://github.com/sgl-project/sglang/pull/21030) |
| 作者 | merrymercy |
| 合并日期 | 2026-03-21 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
流式积压时多个 chunk 共享同一个可变 logprobs 列表导致串块;在 meta_info 记录不可变的长度快照并按区间切片修复。

## 问题现象
`test_completion_stream` 在 CI 上偶发(flaky)失败,日志报 `IndexError: list index out of range`,位置在 `test_openai_server.py` 的 `response.choices[0].logprobs.tokens[0]`(附 CI 日志链接)。触发条件:多个流式 chunk 在消费者取走之前排队积压("streaming backlog")时;此时第一个 chunk 会"偷走"全部 logprobs,后续 chunk 拿到空的 `tokens=[]`,测试断言越界。

## 根因
正文披露:当多个流式 chunk 积压时,所有 chunk 的 `meta_info["output_token_logprobs"]` 指向 `tokenizer_manager.py` 中**同一个可变列表**;后面的 chunk 继续往该列表 extend,于是前面的 chunk 看到属于后面 chunk 的 logprobs——第一个 chunk 偷走全部 logprobs,后续 chunk 的 tokens 为空,导致 `IndexError`。

另外正文指出:PR #17687 的 workaround 只处理了 `finish_reason` 的流结束场景,漏掉了流中途积压的情形。

## 修复方案
1. `python/sglang/srt/managers/tokenizer_manager.py` 的 `add_logprob_to_meta_info`:在创建 chunk 时记录不可变的 int 长度快照:

```diff
         meta_info["output_token_logprobs"] = state.output_token_logprobs
+        meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)
```

2. `serving_chat.py` / `serving_completions.py`:流式处理改用该快照 `total_output_logprobs`,切片从 `[n_prev_token:]` 改为 `[n_prev_token:total_output_logprobs]`,使每个 chunk 只看到属于自己的 logprobs,不受后续 extend 影响;`_process_streaming_logprobs` 增加 `total_output_logprobs` 参数。
3. 回滚 PR #17687 在 `test_openai_server.py` 中的宽松化改动,恢复严格断言(`assert response.choices[0].logprobs` 等)。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围;涉及 #17687 workaround 合入后的版本。
- 回归面:OpenAI 兼容流式接口(chat/completions)的 logprobs 切片逻辑;非流式 logprobs 端点不受影响(它们使用最终累计列表,正文明确)。若上游还有其他消费者直接读 `output_token_logprobs` 列表尾部,行为与旧结果不同。测试计划要求 `test_completion_stream` 在 Blackwell 上稳定通过、`test_chat_completion_stream` 继续通过。

## 关联
- [PR #17687](https://github.com/sgl-project/sglang/pull/17687)(正文提到被回滚的 workaround)
- CI 日志:https://github.com/sgl-project/sglang/actions/runs/23338124262/job/67887146436?pr=20999

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
