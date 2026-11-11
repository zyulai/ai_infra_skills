# PR #44047: [Fix] openai: Validate sampling bounds and preserve chat output limits

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#44047](https://github.com/sgl-project/sglang/pull/44047) |
| 作者 | TheDuyIT |
| 合并日期 | 2026-11-10 |
| labels | documentation, run-ci, run-ci-extra, run-ci-nv-extra |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Sampler / Detokenization |
| bug 类型 | general |

## 一句话结论
多个非法采样值穿透解析后在 serving/scheduler 崩溃或输出垃圾;新增准入校验,并修复 chat 输出上限显式 0 被回退忽略的问题。

## 问题现象
多个非法采样值能通过请求解析,只在 serving 或 scheduler 阶段才失败(chat 输出上限处理原用 `max_completion_tokens or max_tokens`,显式 0 会回退到旧上限)。live server(Qwen2.5-0.5B-Instruct,vocab 151936)观察:
- 巨大 `n` 在请求被展开 n 倍后才被拒:`/generate` `n=3,000,000` 耗时约 113 s、内存涨约 18 GB 才返回 400;main 上 `n=300,000` 15 分钟未完成、内存超 90 GB;`n=0` 无明确报错。
- `top_logprobs` 超过词表(如 200000)用 `torch.topk` 越界杀死服务器;list 或负 `top_logprobs_num` 搞乱一个 batch;logprobs 关闭的请求也能崩——同 prefill batch 里任一请求要 logprobs 时 scheduler 会对每个请求的 count 取 topk。
- 大 `logit_bias` 在 float32 bias 张量溢出为 inf,或 sampler 除以小于 1 的 temperature 后 NaN softmax:`1e39` bias 或 `temperature=0.5` 的 `3e38` 返回垃圾 token(`[0, 0, 0, 0]`),加 `top_p` 则杀死 scheduler。
- 词表外 stop id(如 `-1`、`999999`)在 `min_new_tokens` penalizer 运行时触发 CUDA device-side assert 杀死服务器;penalizer 覆盖整批,stop id 与 `min_new_tokens` 可来自不同请求;beam search 下 `2**63` 到 `2**64-1` 的 stop id 令 scheduler int64 张量构建失败。
- 显式 `max_completion_tokens=0` 被忽略。

## 根因
正文披露的机制:这些值此前缺少准入校验(无 bounds 检查即进入采样路径),以及 chat 上限的 `or` 回退写法把显式 0 当作 falsy 回退到 `max_tokens`。正文未指认引入缺陷的具体提交。

## 修复方案
- 新增 `sampling_params.validate_n`:要求 int(bool 拒绝)、≥1、≤ `SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ`(默认 4096,Rust server 对 n 的同款上限);负值禁用上限。`GenerateReqInput._handle_parallel_sampling` 在任何输入被乘 n 之前调用(dict 与 list 均覆盖),`SamplingParams.verify` 作兜底。
- chat/completions 协议模型保留 OpenAI 上限 `OPENAI_MAX_N = 128`,叠加 `validate_n` 后取两者较小值。
- `tokenizer_manager` 新增 `_validate_top_logprobs_num`:请求 logprobs 时 `top_logprobs_num` 必须是 `[0, vocab_size]` 内整数;logprobs 关闭时该 count 不检查并转发为 0,进不了共享 topk。
- `SamplingParams.verify` 校验 `logit_bias` 值(数字、有限、`|bias| <= 1e30`;sampler 除以 ≥1e-6 的 temperature,该界保证 float32 bias 有限)。
- `min_new_tokens` penalizer 与 beam search 在建张量前丢弃 `[0, vocab_size)` 之外的 stop id(此类 id 永不被采样、按原语义永不匹配)。
- chat `top_logprobs`/`temperature` 拒绝负值,`temperature` 拒绝 inf/nan;`ChatCompletionRequest.effective_max_tokens` 让 `max_completion_tokens` 优先(含显式 0),chat 转换与上下文长度检查都改用它;`max_tokens`/`max_completion_tokens` 必须非负且非布尔。OpenAI 的 temperature 0-2、top_logprobs ≤20、logit_bias [-100,100] 等限制刻意不强制(引擎按值正确执行,现有客户端依赖)。文档同步更新。

关键摘录(`sampling_params.py`):

```diff
+def validate_n(n: Any) -> int:
+    # A negative limit disables the cap, matching the Rust server.
+    max_n = envs.SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ.get()
+    if type(n) is not int or n < 1 or 0 <= max_n < n:
+        bound = f"in [1, {max_n}]" if max_n >= 0 else ">= 1"
+        raise ValueError(f"n must be an integer {bound}, got {n!r}.")
+    return n
```

验证:单 H200 + Qwen2.5-0.5B-Instruct live server 请求级测试,main 上 `n=300000` 无响应服务器被杀,本 PR `n=3000000` 约 3 ms 返回 400;`top_logprobs=200000`、词表外 stop id、越界 `logit_bias` 等用例全部从「服务器死亡/垃圾输出」变为 400 或正常忽略;单元测试每个 guard 均做 mutation 检查。

## 影响与回归风险
- 影响窗口:正文未给出明确版本/提交范围(测试对照 main e109bc1832、f59cc7e45f、d213124956)。
- 回归面:正文行为变化表——`max_completion_tokens=0` 从跑默认上限变为 0 tokens、`finish_reason=length`;`n>128` 在 chat/completions 400;`n>4096` 在 `/generate`、`Engine.generate` 展开前 400;`logit_bias` |bias|>1e30 → 400;词表外 stop id 忽略但不再崩溃;native `n` 129-4096 行为不变,可用 `SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ` 调整。已知限制(正文):opt-in Rust HTTP server 不走 tokenizer_manager,这些准入检查不适用(stop-id 过滤在 scheduler 端仍覆盖);`batch_size * n` 仍无上界,`max_tokens` 无上界;speculative decoding 下的零长度生成未测。注:pack 的 diff 在 40000 字符处截断(文件粒度),截断部分未读,可能还有未列入本卡的其余测试文件改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
