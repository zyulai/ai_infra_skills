# PR #28670: [ROCm][Qwen3-32B] Fix AITER MHA accuracy issue cause by #25763

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28670](https://github.com/vllm-project/vllm/pull/28670) |
| 作者 | sammysun0711 |
| 合并日期 | 2025-11-17 |
| labels | rocm, ready, v1, qwen |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
AITER MHA 在 prefill/extend 传入错误 min_seqlen_q 触发 ck 前置检查直接跳过内核,固定传 1 修复精度损失。

## 问题现象
ROCm 上 Qwen3-32B 使用 AITER MHA 后端出现精度下降(issue #28598 报告)。作者在 MI308X 上用 lm-eval gsm8k(5-shot)复现:修复前 flexible-extract exact_match 0.4769、strict-match 0.5428;而默认 Triton Unified Attention 分别为 0.6300 / 0.7392,差距明显。触发条件是 `VLLM_ROCM_USE_AITER=1` 且走 AITER MHA 后端的纯 prefill 与 extend 阶段,影响面为该后端下所有经过分阶段 attention 路径的请求输出质量。

## 根因
正文披露:#25763 重构 AITER MHA 后端(默认 AITER 版本 `9716b1b8`)时把 attention 拆成 pure prefill / extend / decode 三阶段,`min_seq_len` 改由 `attn_metadata.prefill_metadata.min_query_len` 与 `attn_metadata.extend_metadata.min_query_len` 传入;而重构前 prefill 阶段是硬编码 `min_seq_len=1`。错误的 `min_seq_len` 传给 `aiter.flash_attn_varlen_func` 后,会命中 composable_kernel `fmha_fwd_pagedkv_kernel` 的前置条件检查,该检查直接 return、不执行内核,注意力输出因此不正确,表现为精度下降(无报错、静默出错)。

## 修复方案
改 `vllm/v1/attention/backends/rocm_aiter_fa.py` 的 `forward`,在纯 prefill 与 extend 两处调用 `aiter.flash_attn_varlen_func` 时把 `min_seqlen_q` 恢复为硬编码 1,不再依赖 metadata 的 min_query_len:

```diff
-                    min_seqlen_q=attn_metadata.prefill_metadata.min_query_len,
+                    min_seqlen_q=1,
```

extend 阶段同样由 `attn_metadata.extend_metadata.min_query_len` 改为 `1`。共两行改动,decode 路径未动。

验证:MI308X、镜像 rocm/vllm-dev/nightly_main_20251113、`VLLM_ROCM_USE_AITER=1` + tp 2 起 Qwen/Qwen3-32B,lm_eval gsm8k 修复后 flexible-extract 0.6293 / strict-match 0.7210,与 Triton 默认后端(0.6300 / 0.7392)基本持平,确认精度恢复。

## 影响与回归风险
- 影响窗口:自 #25763 重构 AITER MHA 后端合入起,至本修复前,使用 AITER MHA 后端的版本均受影响(正文指认该 PR 引入)。
- 回归面:仅 ROCm AITER FA 后端 `rocm_aiter_fa.py` 的 prefill/extend 两条调用路径;改动是把 `min_seqlen_q` 恢复为重构前的既有语义,decode 与其他 attention 后端不受影响。风险较低,但若后续 metadata 的 min_query_len 语义另有用途,此处不再消费它。

## 关联
- [#28598](https://github.com/vllm-project/vllm/issues/28598)(精度问题报告)
- [#25763](https://github.com/vllm-project/vllm/pull/25763)(引入问题的重构)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
