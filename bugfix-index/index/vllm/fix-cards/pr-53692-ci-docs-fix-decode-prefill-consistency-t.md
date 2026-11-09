# PR #53692: [CI][Docs] Fix decode/prefill consistency test prefix construction; add GLM-4-9B and Phi-4 to batch-invariant tested models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53692](https://github.com/vllm-project/vllm/pull/53692) |
| 作者 | LioEinaudi |
| 合并日期 | 2026-10-04 |
| labels | documentation, ready, glm |
| 热度 | 👍 2 · 💬 31 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
batch invariance 的 decode/prefill 一致性测试用「文本前缀重新编码」构造 prefill 输入,对 BPE 词表不无损,产生假失配;改为直接用 token ids 构造前缀,并把 GLM-4-9B、Phi-4 加入已验证模型文档。

## 问题现象
`tests/v1/determinism/test_batch_invariance.py::test_decode_logprobs_match_prefill_logprobs` 对部分模型报假的 decode/prefill logprobs 失配。典型例子:GLM-4-9B 在官方套件下报告 mismatch——模型把 `'.'` + `'Here'` 编码为 `[13, 8419]`,而该文本重新编码会得到单个 token `89329`(`'.Here'`),文本前缀从位置 11 起偏离。任何 greedy 输出命中 BPE merge 点的模型都会以同样方式假失败。修复前 GLM 在 H20 上 18/19、在 24 GB 4090D 上 12/19(6 个 needle 用例 init OOM + 1 个本假失配);修复后 GLM 各平台 19/19 全绿。

## 根因
PR 正文披露:该测试此前把每个 prefill 前缀从文本重建再重新编码,而 `encode(decode(ids))` 对 BPE tokenizers 不是无损操作,于是「prefill」评估的 token 序列可能与 decode 时的不同,报告假失配。这是测试本身的健全性(soundness)bug,非推理引擎问题。此修复来自 #43317(@Evangade),原报告为 #27433 的 @xRay2016;#43317 自 5 月 review 后无活动被标记 stale,应维护者要求由本 PR 携带其 commit 与原署名。

## 修复方案
1. `tests/v1/determinism/test_batch_invariance.py`:前缀改为直接由 token ids 构造 `prefix_token_ids = prompt_token_ids + list(token_ids[:token_idx])`,prefill 用 `TokensPrompt(prompt_token_ids=prefix_token_ids)` 发起;同时删掉原先每个 decoded token 都要额外跑一次 `llm.generate` 重建前缀文本的做法(消除 O(N²) 的额外 generate),失败诊断输出改为打印前缀 token ids;
2. `docs/features/batch_invariance.md`:已验证模型清单加入 `zai-org/glm-4-9b-chat`(GLM-4 series)与 `microsoft/phi-4`(Phi series),后者按 #53247 的「无需单独 issue 提交模型验证」邀请补充。

关键 diff 摘录:

```diff
-            if token_idx == 0:
-                prefix_prompt = prompt
-            else:
-                # ... regenerate with max_tokens = token_idx
-                prefix_output = llm.generate([prompt], prefix_sp, use_tqdm=False)[0]
-                prefix_prompt = prompt + prefix_output.outputs[0].text
+            prefix_token_ids = prompt_token_ids + list(token_ids[:token_idx])
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:改动限于确定性测试文件与文档,不影响任何运行时代码路径。测试语义变化:prefill 输入从「前缀文本」变为「前缀 token ids」,验证的是真正的同 token 序列;#43317 中的独立 tokenizer 回归测试按其 review 意见未携带。验证覆盖:RTX 4090 D(sm89, v0.28.0)、H20/H100 NVL(sm90)、RTX PRO 6000 Blackwell(sm120, v0.29.0 cu130)——修复后默认模型与 Phi-4、GLM-4-9B 均 19/19 通过,sm120 为套件首次运行且无 flaky 重试。

## 关联
- #43317(被携带的修复,已 stale): https://github.com/vllm-project/vllm/pull/43317
- #27433(原始报告): https://github.com/vllm-project/vllm/issues/27433
- #53247(模型验证提交邀请的评论): https://github.com/vllm-project/vllm/pull/53247#issuecomment-5374419058
- @yuvalluria 的 H100 NVL 运行结果(Phi-4 / GLM): https://github.com/vllm-project/vllm/pull/53692#issuecomment-5618510438

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
