# PR #23829: [bugfix] [spec-decoding] fix data race in sample_recovered_tokens_kernel (vLLM v1)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23829](https://github.com/vllm-project/vllm/pull/23829) |
| 作者 | He-Jingkai |
| 合并日期 | 2025-08-28 |
| labels | ready, v1 |
| 热度 | 👍 7 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
`sample_recovered_tokens_kernel` 用「写 0 再恢复」的方式清零 draft token 概率,多 GPU 线程并发访问产生 data race;改为 tl.load 时用 mask 直接置 0。

## 问题现象
vLLM V1 speculative sampling 开发中发现:该 data race 导致不同 TP(张量并行)GPU 产生不一致的 speculative decoding 验证结果;一旦验证结果不一致,GPU 就会 hang(参见 #17929)。表现为非确定性行为与正确性问题,TP>=2 且开启 spec decoding 时触发。作者在 Qwen2.5-7B/14B/32B、Qwen3-8B/14B 上以 TP>=2 验证修复。

## 根因
正文披露:`sample_recovered_tokens_kernel`(Triton kernel)原实现在 `NO_DRAFT_PROBS`(如 n-gram,无 draft_prob,视为 1)分支中,临时把 `target_probs_ptr` 中 draft token 的概率写 0,算完后再恢复原值。在多线程(多 GPU 线程)环境下,多个线程会并发访问并修改同一内存地址,读-改-写序列交错导致 data race,产生非确定性结果。

## 修复方案
改 `vllm/v1/sample/rejection_sampler.py`:删除「先写 0、算完恢复 `orig_prob`」的读-改-写模式,改为在 `tl.load` 的 mask 上直接排除 draft token,加载时该位置即为 0,不再修改内存:

```diff
-        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
-                            draft_token_id)
-        tl.store(
-            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
-            0)
         prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                        vocab_offset,
-                       mask=vocab_offset < vocab_size,
+                       mask=((vocab_offset < vocab_size) &
+                             (vocab_offset != draft_token_id)),
                        other=0)
```

逻辑意图不变(target_prob - draft_prob 的等价形式),同时可能带来性能提升。作者称该修改在内部 HistoSpec RL 项目中经数千 GPU 小时使用,修复后 TP>=2 spec decoding 正常,跨 TP 验证不一致导致的 GPU hang 消失,无性能退化。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:`vllm/v1/sample/rejection_sampler.py` 中该 Triton kernel 的 `NO_DRAFT_PROBS` 分支(n-gram 等 draft 概率为 1 的推测解码路径);其余分支与正常采样路径不变。mask 方式语义等价,主要风险在边界条件(vocab_size 对齐等)上的行为差异。

## 关联
- #17929(验证结果不一致导致 GPU hang 的相关 PR)
- HistoSpec RL 项目论文(https://www.alphaxiv.org/abs/2508.18588v1,bug 发现场景)
- 正文未列出其他 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
