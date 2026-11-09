# PR #28445: [Bugfix] fix kimi-linear crash

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28445](https://github.com/vllm-project/vllm/pull/28445) |
| 作者 | ZJY0516 |
| 合并日期 | 2025-11-13 |
| labels | ready |
| 热度 | 👍 1 · 💬 11 |
| 分类 | Model-specific Fast Path |
| bug 类型 | crash |

## 一句话结论
Kimi-Linear 的 KDA 层送入 kernel 的张量未按实际 token 数截断、索引长度误用,导致 FULL_AND_PIECEWISE 模式下 IMA 崩溃;修复为按 `num_actual_tokens` 截断并修正 cu_seqlens/索引。

## 问题现象
对应 issue #28436 报告的 Kimi-Linear 崩溃。作者在正文 TODO 中补充了一个未解决的相关现象:`tp=4, dp=2` 时仍会崩溃,IMA(illegal memory access)发生在 `fused_recurrent_gated_delta_rule_fwd_kernel` 中,且仅在 `FULL_AND_PIECEWISE` 调度模式下出现。验证方式(Test Plan):`vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct --enable-expert-parallel -tp 8 --trust-remote-code`,再用 lm_eval 跑 gsm8k(5-shot,280 并发);Test Result:flexible-extract exact_match 0.8901±0.0086,strict-match exact_match 0.8779±0.0090。

## 根因
正文未系统披露根因,仅给出修复目的与上述 TODO 现象。(由 diff 推断)改动显示问题机制:进入 `kda_attention` 的 `q/k/v_proj_states`、`g1`、`beta` 未按 `attn_metadata.num_actual_tokens` 截断就送入 kernel;decode 分支的 `decode_conv_indices` 用 `attn_metadata.num_decodes` 去截取 `non_spec_state_indices_tensor`,而 `cu_seqlens=non_spec_query_start_loc` 也未按 `num_decodes + 1` 截断,且输出用整体赋值 `core_attn_out[:] = core_attn_out_non_spec` 并附带 shape assert。张量长度与实际 token 数、索引长度互相不一致,导致 `fused_recurrent_gated_delta_rule_fwd_kernel` 内越界访问(IMA)。

## 修复方案
改动仅 `vllm/model_executor/layers/kda.py`:
- 删除未使用的 `g2` 参数(`kda_attention`/`kda_attention_fake`/`forward`/`_forward` 的签名与调用点同步移除);
- `_forward` 中取 `num_actual_tokens = attn_metadata.num_actual_tokens`,对 `q_proj_states/k_proj_states/v_proj_states/g1/beta` 执行 `[:num_actual_tokens]` 截断;
- decode 分支索引改为 `non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens]`;
- kernel 调用改 `cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1]`;
- 输出改逐段赋值并删掉 shape assert:

```diff
+        q_proj_states = q_proj_states[:num_actual_tokens]
+        k_proj_states = k_proj_states[:num_actual_tokens]
+        v_proj_states = v_proj_states[:num_actual_tokens]
+        g1 = g1[:num_actual_tokens]
+        beta = beta[:num_actual_tokens]
...
-                cu_seqlens=non_spec_query_start_loc,
+                cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
...
-        assert core_attn_out_non_spec.shape == core_attn_out.shape
-        core_attn_out[:] = core_attn_out_non_spec
+        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
+            0, :num_actual_tokens
+        ]
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:仅 `kda.py` 的 KDA 注意力路径(Kimi-Linear 模型);非 spec decode 分支的截断与索引逻辑变化会影响 FULL_AND_PIECEWISE 等调度模式下该 kernel 的输入长度;`g2` 参数删除会改变自定义调用方的签名兼容性。作者明确注明 `tp=4, dp=2` 下的 IMA 本 PR 未解决(正文 TODO)。

## 关联
- FIX https://github.com/vllm-project/vllm/issues/28436

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
