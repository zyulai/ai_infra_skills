# PR #45630: [Perf] dsv41: Use tuned MXFP8 tactics at decode sizes

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kpham-sgl |
| 合并 | 2026-10-21 |
| 分类 | Quantization (tier: major) |
| Label | deepseek |
| Diff | pack 未提供 diffstat（diff 内容可用：2 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/45630 |

## 一句话结论

DeepSeek-V4.1 把 block-fp8 MXFP8 linears 的 tuned tactic 启用门槛从 4096 行降到 1 行，128K 上下文 decode step 快 2-3%。

## 动机

block-fp8 MXFP8 linears 启动时对每个 M bucket 做 autotune，但只有达到 4096 行才使用调优出的 tactic，更小的 M（即 decode 全部请求）始终跑被钉死的默认 tile——autotune 结果在 decode 路径上白费。

## 优化点

`deepseek_v4.py` 的 `autotune_prefill_kernels` 中，`block_fp8_as_mxfp8` 时把 `mxfp8_prefill_autotune_min_tokens` 从 4096 降为 1，即 autotune 出的 tactic 在任意行数（含 decode 小 M）都可被 dispatch 采用。两个例外保持钉死默认 tactic：deterministic 与 batch-invariant 执行（原文注释从 "Small shapes and deterministic execution keep their pinned tactic" 改为 "Deterministic execution keeps its pinned tactic"，小 shape 钉死取消）。改动仅一行阈值 + 单测更新。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: autotune_prefill_kernels 里 mxfp8_prefill_autotune_min_tokens 4096 -> 1
test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py: 期望值改 1；删除 test_prefill_tuning_leaves_decode_bit_identical（decode 不再位级保持旧 tactic）
```

## 性能声明

DEP8 decode，128K 上下文：step 在每 rank 16-64 请求时快 2%、256 请求时快 3%。硬件、模型并行度等其余测试条件 PR 未注明。

## 验证与回归风险

单测更新：`test_model_hook_deduplicates_ready_block_fp8_weights` 期望 min_tokens=1；`test_block_fp8_dispatch_keeps_decode_and_determinism_pinned` 继续验证 decode/determinism 的钉死行为；原 `test_prefill_tuning_leaves_decode_bit_identical`（断言 prefill tuning 前后 decode 输出位级一致）被删除——因为 decode tactic 现在会变，该位级保证不再成立，意味着 MXFP8 decode 输出数值与旧版可能有微小差异（PR 未给 e2e 精度数据）。CI 状态合并时显示 Base/Extra/AMD 均为 x（失败），PR 正文未说明。

## 关联

PR 正文未提及关联 PR；同区域为 DeepSeek-V4.1 的 MXFP8/FlashInfer cutedsl GEMM 路径其他优化（参见索引同分类条目）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
