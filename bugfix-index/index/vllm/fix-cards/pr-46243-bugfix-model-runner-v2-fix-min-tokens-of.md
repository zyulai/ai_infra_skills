# PR #46243: [Bugfix][Model Runner V2] Fix min_tokens off-by-one in the V2 GPU sampler

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46243](https://github.com/vllm-project/vllm/pull/46243) |
| 作者 | Sunt-ing |
| 合并日期 | 2026-06-21 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
V2 GPU 采样器 min_tokens 出 off-by-one:EOS 晚一步放行,多生成 1 个非 EOS token;kernel 中停止条件改为 `pos + 1 < min_len` 对齐 V1 行为。

## 问题现象
设置 `min_tokens=N` 时,EOS 应在第 N 个输出(即第 N+1 个 token)被放行,与 V1 的 `MinTokensLogitsProcessor` 一致;但 V2 GPU 采样器晚一步放行,导致 `min_tokens=N` 实际强制生成 N+1 个非 EOS token。对 `min_tokens>=1` 的所有请求都多出 1 个 token(实测表:V2 main 在 min_tokens=1/2/4 时生成长度为 3/4/6,V1 参考为 2/3/5)。该 kernel 是 Llama、Qwen3、Mistral 等主流架构的默认路径。

## 根因
正文披露:`vllm/v1/worker/gpu/sample/logit_bias.py` 中的 Triton kernel 在 `pos < min_len` 时抑制 stop token,但 `pos` 是最后一个 token 的位置(当前长度减一),因此抑制持续到"当前长度已达 min_len 之后"才结束,晚了一步。

## 修复方案
改 kernel 中的停止条件,改为比较当前长度:

```diff
-    if num_stop_token_ids > 0 and pos < min_len:
+    if num_stop_token_ids > 0 and pos + 1 < min_len:
```

`min_tokens=0` 不受影响(仍由 `num_stop_token_ids > 0` 守卫)。验证方式:通过 `logit_bias` 强制 EOS(temperature=0、min_tokens=N、max_tokens=32、logit_bias={eos:100.0}),在 RTX 4090 + Qwen/Qwen3-0.6B 上对比生成长度——修复后 V2 与 V1 完全一致(1/2/3/5 对应 min_tokens=0/1/2/4)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 V2 GPU 采样器的 stop-token 抑制 kernel(`logit_bias.py::_bias_kernel`)这一处条件;`min_tokens>=1` 的生成提前一步放行 EOS,`min_tokens=0` 与无 stop token 场景行为不变(由 diff 推断)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
