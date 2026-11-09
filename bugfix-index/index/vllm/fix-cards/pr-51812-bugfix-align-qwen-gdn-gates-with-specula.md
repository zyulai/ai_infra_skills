# PR #51812: [Bugfix] Align Qwen GDN gates with speculative tokens

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51812](https://github.com/vllm-project/vllm/pull/51812) |
| 作者 | ZJY0516 |
| 合并日期 | 2026-08-11 |
| labels | bug, qwen |
| 热度 | 👍 6 · 💬 6 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
混合批次中非投机 token 排在投机 token 前,QKV 已按索引 gather 而 a/b 门控未 gather,行错位;对门控用相同索引 gather 修复。

## 问题现象
Qwen GDN 投机解码在"混合批次把非投机 token 排在投机 token 之前"时出错。正文用 `Qwen/Qwen3.5-2B`(BF16、TP1、V1 runner、eager 模式、2 个 MTP draft tokens、`max_model_len=128`)端到端复现:小的 max_model_len 让边界条件很快到达(任意配置的 context 上限附近都可能发生);120/64 token 的 prompt 产生了"一行单 token 非投机行在三 token 投机行之前"的混合步,运行时元数据为 `spec_token_indx=(1,2,3)`、`non_spec_token_indx=(0,)`。与 no-MTP 目标模型参考从混合步后首个 token 起对比:chosen-logprob 平均绝对误差 0.002755 → 0.000208,最大误差 0.020539 → 0.001690;本例中贪心 token ID 前后一致(logit 漂移未跨过 argmax 边界)。V2 runner 上投机 token 排在前面,本就不受影响。正文还披露混合布局的成因:GDN metadata builder 只按 query 长度把单 token 非投机行归为 decode、不看请求是否仍在 prefill;存在投机 decode 时该行被重分类到 prefill kernel,但在 token 拆分上仍属非投机——真实单 token decode 或最后一段单 token prefill chunk 都会形成这种布局。

## 根因
正文披露:`mixed_qkv` 用 `spec_token_indx` gather,但 fused recurrent update 收到的仍是未排序的 `a`、`b` 门控张量;kernel 只消费前 `T_spec` 行门控,于是门控可能与 gather 后的 Q/K/V 行属于不同 token。

## 修复方案
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 的 `_forward_core`:混合路径下对 `a`、`b` 用与 `mixed_qkv` 相同的索引 gather,并把 `a_spec`/`b_spec` 传给 `fused_sigmoid_gating_delta_rule_update`;纯投机快速路径(`num_prefills == 0 and num_decodes == 0`)保持 `a_spec = a, b_spec = b` 不变:

```diff
                 mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
+                a_spec = a.index_select(0, spec_token_indx)
+                b_spec = b.index_select(0, spec_token_indx)
                 mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
...
-                    a=a,
-                    b=b,
+                    a=a_spec,
+                    b=b_spec,
```

## 影响与回归风险
- 影响窗口:正文未说明(触发条件为 V1 runner + 混合批次且非投机行位于投机行之前,常见于接近 context 上限处)。
- 回归面:仅 `qwen_gdn_linear_attn.py` `_forward_core` 中投机路径的门控输入来源;纯投机快速路径与非投机路径逻辑不变。验证:`tests/kernels/mamba/test_gdn_forward_core_split.py` 8 passed;pre-commit(Ruff、mypy)通过;Qwen 3.5 Blackwell CI(buildkite 构建 83349)通过。PR 由 OpenAI Codex 协助完成,提交者声明已逐行复核。

## 关联
- buildkite 测试通过链接:https://buildkite.com/vllm/ci/builds/83349/canvas?jid=019ff0e3-ac1a-4a4e-a797-ef903d9c7985&tab=output
- 其余 issue/PR:正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
