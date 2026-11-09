# PR #12115: Fix Illegal Instruction/IMA errors when using DP attention -- num_tokens_for_logprob calculation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12115](https://github.com/sgl-project/sglang/pull/12115) |
| 作者 | YAMY1234 |
| 合并日期 | 2025-10-25 |
| labels | run-ci |
| 热度 | 👍 5 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
--enable-dp-attention 下 num_tokens_for_logprob 按"全部 token 需 logits"计算,与剪枝后的实际 token 数不符致 DP gather 越界;return_logprob=False 时改用 batch_size()。

## 问题现象
使用 `--enable-dp-attention` 时,logits 阶段的 DP gather 使用错误 metadata,Triton kernel 越界访问,报错(正文照抄):`AssertionError: src OOB: sz=661, chunk_size=7168, required=4738048, src.numel=21504`。即 DP gather 期望 661 tokens 实际只收到 3 tokens(local_hidden_states.shape=(3,7168) 而 global 侧按 661 准备)。该 PR 是对 #12052 的根修复。

## 根因
正文披露:scheduler 计算 `num_tokens_for_logprob` 时总是假设所有 token 都需要 logits 计算;而当 `return_logprob=False` 时,`LogitsProcessor` 只构造每请求最后一个 token 的 `pruned_states`(batch_size 个 token)。两侧计数不匹配,DP gather 按错误的 token 数同步,造成越界访问。

## 修复方案
`managers/scheduler.py::prepare_mlp_sync_batch_raw()` 中 extend 模式的计算分两种情况:

```python
+            if local_batch.return_logprob:
+                num_tokens_for_logprob = sum(
+                    max(extend_len - logprob_start_len, 1)
+                    for logprob_start_len, extend_len in zip(
+                        local_batch.extend_logprob_start_lens,
+                        local_batch.extend_lens,
+                    )
+                )
+            else:
+                # When return_logprob = False, only need last token per request
+                num_tokens_for_logprob = local_batch.batch_size()
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:仅 DP attention 下 extend 模式 `num_tokens_for_logprob` 的统计口径;`return_logprob=True` 保留原逻辑,DP 各 rank 间同步的 token 数从此与 `LogitsProcessor` 的 pruned_states 实际大小一致。正文附 gsm8k 精度验证:Repeat 8, mean 0.795。

## 关联
- https://github.com/sgl-project/sglang/pull/12052(本 PR 为其根修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
