# PR #14568: permute/unpermute kernel for moe optimization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @CalebDu |
| 合并 | 2025-05-02 |
| 分类 | MoE \| tier: major |
| Label | ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 12 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/14568 |

## 一句话结论

新增 moe_permute / moe_unpermute kernel，为 MoE grouped-GEMM 整理并重排 token。

## 动机

PR 正文说明 MoE 计算中各 expert 对应的 token 在 activation 里不连续，需要先按 expert 聚集才能调用 grouped-GEMM 加速；相应地 grouped-GEMM 的输出是展开（expanded）形式，需要归约并按 `topk_weight` 缩放回原 token 顺序。实现参考了归档仓库 `BBuf/tensorrt-llm-moe` 中 tensorrt-llm 的 MoE kernel。

## 优化点

`moe_permute` kernel 把 activation 中的 token 展开并按 expert 排序，为 grouped-GEMM 准备连续输入；`moe_unpermute` kernel 归约展开后的 grouped-GEMM 输出并乘以 `topk_weight`。diff 中 `fused_topk` 返回值新增 `token_expert_indices`，供 permute 记录 token 与 expert 的映射，`benchmark_grouped_gemm_cutlass.py` 与 `benchmark_moe.py` 同步适配。新增 benchmark 用 CUDA graph 捕获 10 次调用取平均（us），对比 customized permute 与 deep_gemm 的 `_moe_permute` 路径，覆盖 fp8_w8a8 / int8_w8a16，并在 fp8 下用 align_block_size=128（deepgemm 需要 128 m 对齐）。正文明确当前不支持带 `expert_map` 的 Expert-Parallelism，将后续更新。

## 关键改动

```text
csrc/moe/moe_permute_unpermute_op.cu: 新增 permute/unpermute 的 CUDA op 入口
csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu/.h/.inl: kernel 实现与声明
csrc/moe/permute_unpermute_kernels/dispatch.h: 按 dtype / 路径分派
benchmarks/kernels/benchmark_moe_permute_unpermute.py: 新增 349 行 permute/unpermute 计时 benchmark
benchmarks/kernels/benchmark_moe.py: fused_topk 调用适配新增的 token_expert_indices
```

分别摘自 benchmark_grouped_gemm_cutlass.py 与 benchmark_moe_permute_unpermute.py 的 diff：

```diff
-    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)
+    topk_weights, topk_ids, token_expert_indices = fused_topk(
+        a, score, topk, renormalize=False)
+            (permuted_hidden_states, first_token_off, inv_perm_idx,
+             m_indices) = moe_permute(
+                 qhidden_states, topk_ids=topk_ids,
+                 token_expert_indices=token_expert_indices,
```

## 性能声明

PR 正文未给出定量性能数据：正文只描述 kernel 机制并附一张示意图，未列出加速比、延迟或吞吐数字，也没有模型 / 硬件 / 精度 / workload 等测试条件，故测试条件 PR 未注明；pack 的 perf claims 检测同为 "(none detected by regex)"。

## 验证与回归风险

PR 未注明 Test Plan 与 Test Result，也未提供 CI 结果；可依赖的验证手段是新增的 `benchmark_moe_permute_unpermute.py`（对比 customized permute 与 `_moe_permute` 的时延）。回归风险方面，正文明确 "Currently, unsupport Expert-Parallelism with expert_map, will follow up with updates"；`fused_topk` 返回值签名变化要求调用点同步（diff 已改两个 benchmark 文件），其余受影响路径 PR 未注明。

## 关联

无（正文仅提及参考 TensorRT-LLM 的 MoE kernel 实现，未给出同区域 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
