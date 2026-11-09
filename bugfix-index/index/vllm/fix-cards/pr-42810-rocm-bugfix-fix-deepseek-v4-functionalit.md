# PR #42810: [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42810](https://github.com/vllm-project/vllm/pull/42810) |
| 作者 | tjtanaa |
| 合并日期 | 2026-05-17 |
| labels | bug, deepseek, ready, rocm, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 DeepSeek V4 因 #41263 后多调一次 ffn_norm 而功能损坏,且 AITER MHC 精度缺陷与 prefill topk buffer 损坏导致高并发精度下降;回退 torch MHC、修正 topk 调用修复。

## 问题现象
两点:1) #41263 合并后 DeepSeek V4 在 ROCm 上功能直接损坏(functionality broken);2) 高并发下精度退化——正文用 lm-eval gsm8k(256 并发、8-shot)验证,修复后 DeepSeek-V4-Pro flexible-extract/strict-match exact_match 为 0.9553/0.9560,DeepSeek-V4-Flash 为 0.9492/0.9500。

## 根因
正文披露:第 2 点是复合问题:
1. AITER MHC 在 token 数很大时有精度问题(AITER 即将发布修复版,故代码路径保留注释便于日后升级 AITER 后重新启用),现回退到 torch 实现;
2. prefill mqa topk buffer 被错误逻辑损坏,改用 `torch.ops._C.top_k_per_row_prefill` 修复。
第 1 点(功能损坏)由 #41263 引入,修复方式是移除 `x = self.ffn_norm(x)`。

## 修复方案
- `vllm/model_executor/models/deepseek_v4.py`:`_forward_rocm` 中删除 `x = self.ffn_norm(x)`——ffn_norm 已折叠进 `self.ffn.norm_gate`,`ffn()` 直接接收 pre-norm 激活(修复功能损坏)。
- `vllm/model_executor/layers/mhc.py`:`mhc_pre`/`mhc_post` 的 `forward_hip` 注释掉 AITER 路径(`torch.ops.vllm.mhc_pre_aiter`/`mhc_post_aiter`),无条件走 `mhc_kernels.mhc_pre_torch`/`mhc_post_torch`。
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`:prefill/decode topk 改为直接调用 `torch.ops._C.top_k_per_row_prefill` / `top_k_per_row_decode`,删除 `_topk_indices_prefill`/`_topk_indices_decode` 包装;`rocm_aiter_sparse_attn_indexer_native` 合并为正式 op(加 `@eager_break_during_capture`,签名增加 `skip_k_cache_insert`);`_topk_indices_torch` 增加 row_starts 参数以对齐 CUDA kernel 的"索引为每行有效区间局部值"契约。
- `vllm/model_executor/layers/sparse_attn_indexer.py`:删除非 AITER 的 native 回退分支,ROCm 路径要求 `VLLM_ROCM_USE_AITER=1`,否则抛 RuntimeError。

## 影响与回归风险
- 影响窗口:#41263 合并之后、本 PR 之前的版本;正文以 PR 链接为界,未给出具体版本号。
- 回归面:ROCm 上 DeepSeek V4 的 MHC 前后处理(现固定走 torch 实现,大 token 数下可能比 AITER kernel 慢)、sparse attention indexer 的 topk 计算(prefill/decode 索引语义对齐 CUDA kernel 契约)、非 AITER 环境下 sparse_attn_indexer 直接报错不再有回退(由 diff 推断)。

## 关联
- https://github.com/vllm-project/vllm/pull/41263
- https://github.com/ROCm/aiter/commit/b639cb63bcac4672dce33a731fad042a65cb3649 (diff 注释中引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
