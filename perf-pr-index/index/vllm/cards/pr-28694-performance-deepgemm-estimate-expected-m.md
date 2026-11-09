# PR #28694: [Performance][DeepGEMM] Estimate expected_m

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @varun-sundar-rabindranath |
| 合并 | 2025-11-15 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | performance, ready, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/28694 |

## 一句话结论

按均匀分布估计 DeepGEMM 的 expected_m（向上取整到 16 的倍数），性能最多提升 11%。

## 动机

DeepGEMM 的 `fp8_m_grouped_gemm_nt_masked` kernel 接收 `expected_m` 参数，是给 kernel 的「每专家 token 数估计」hint；main 上直接把它设为专家可拥有的最大 token 数。本 PR 更新该估计逻辑：假设均匀分布，把 M 的期望值向上取整到下一个 16 的倍数（正文指向外部 Google Sheets 的 "Estimate_m step fn"，给出不同 cuda graph sizes 与 DP sizes 下的 expected_m 阶跃函数）。

## 优化点

在 vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py 新增 `estimate_expected_m(global_num_experts, max_tokens_per_expert, topk)`：从 forward context 取 dp_metadata（不可用时 warning 并回退 `max_tokens_per_expert`），取 `total_num_tokens = dp_metadata.num_tokens_across_dp_cpu.sum()`、`total_num_tokens_replicated = total_num_tokens * topk`，按均匀负载均衡假设算 `round_up(total_num_tokens_replicated // global_num_experts, 16)`，再 clamp 到 [16, max_tokens_per_expert]。`apply()` 里原本 `expected_m = max_num_tokens` 改为调用该函数。vllm/forward_context.py 新增 `is_forward_context_available()`；测试用 `with_dp_metadata` 构造 forward context 并包裹 `mk.forward`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py: 新增 estimate_expected_m 并在 apply() 中替换 expected_m 赋值
vllm/forward_context.py: 新增 is_forward_context_available()
tests/kernels/moe/test_deepep_deepgemm_moe.py: 新增 with_dp_metadata 上下文管理器
```

```python
        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        # Assume even load balancing
        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        # clamp estimate
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate
```

## 性能声明

PR 正文 Performance 段仅给出 "TLDR; The performance improvements upto 11%"（perf claims 亦为 "performance improvements upto 11%"），并把不同 cuda graph sizes / DP sizes 下的 expected_m 阶跃函数与详细数字指向外部 Google Sheets（round-16），正文内未提供 benchmark 表格；PR 未注明该 11% 的测试条件。Test Plan 另给出服务配置（VLLM_ALL2ALL_BACKEND=deepep_low_latency、VLLM_USE_DEEP_GEMM=1、Qwen/Qwen3-30B-A3B-FP8、--tensor-parallel-size 1 --data-parallel-size 2 --enable-expert-parallel --no-enable-prefix-caching）与 gsm8k lm-eval 命令，但正文未把这些与 11% 数字绑定为测试条件。

## 验证与回归风险

Test Plan：`canhazgpu run -g2 -- vllm serve Qwen/Qwen3-30B-A3B-FP8 ...` 与 `lm_eval --model local-completions --tasks gsm8k ... --limit 100`；Test Result：gsm8k 5-shot flexible-extract 0.90 ± 0.0302、strict-match 0.92 ± 0.0273。diff 中 tests/kernels/moe/test_deepep_deepgemm_moe.py 新增 `with_dp_metadata` 并用 `set_forward_context(num_tokens, num_tokens_across_dp=...)` 包裹 forward。回归风险：DPMetadata 不可用时 warning 并回退 `max_tokens_per_expert`；估计基于均匀分布（even load balancing）假设；估计被 clamp 到 [16, max_tokens_per_expert]；影响 DeepGEMM `fp8_m_grouped_gemm_nt_masked` 的 MoE / DP / EP 路径与不同 cuda graph sizes 下的行为。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
