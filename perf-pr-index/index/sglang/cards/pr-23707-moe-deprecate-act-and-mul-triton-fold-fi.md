# PR #23707: [MoE] Deprecate act_and_mul_triton; fold filter_expert into JIT silu/gelu_and_mul

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ch-wan |
| 合并 | 2026-04-26 |
| 分类 | MoE |
| Label | run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见删除 act_and_mul_kernel 等 Triton 代码,改动集中在 jit_kernel 与 fused_moe |
| 链接 | https://github.com/sgl-project/sglang/pull/23707 |

## 一句话结论

删除 MoE 里重复的 Triton activation kernel,把 filter_expert(跳过 routed expert id 为 -1 的行)折入更快的 JIT CUDA silu/gelu_and_mul,EP 下 skip 行时激活耗时随 skip 比例线性下降。

## 动机

`fused_moe_triton_kernels.py` 中的 `act_and_mul_triton` 与已有的 `silu_and_mul` / `gelu_and_mul` 功能重复,唯一差异是支持 `filter_expert=True`(EP 下跳过 routed expert id 为 -1 的行)。而 JIT CUDA 版的 `silu_and_mul` / `gelu_and_mul` 本身已存在且更快。合并后可删除约 100 行 Triton 代码和一个冗余 kernel。

## 优化点

在 JIT activation kernel 的 `ActivationParams` 中加入 `expert_ids`(int32 指针)与 `expert_step`,并给 `act_and_mul_kernel` 增加编译期模板参数 `kFilterExpert`:关闭时 `if constexpr` 直接跳过加载,零开销;打开时每行计算前先查 `expert_ids[token_id / expert_step]`,等于 -1 就提前 return,不再读 gate/up 与做激活。`expert_step` 为 1 表示 per-token 路由,为 `BLOCK_SIZE_M` 表示 sorted/TMA 路由。Python 侧 `run_activation` / `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` 接受可选 `expert_ids`、`expert_step` 参数,旧调用点不变。MoE 调用点把两处 `act_and_mul_triton` 替换为带 filter 参数的 JIT kernel;CUDA 走 JIT filtered 路径,HIP/XPU 因 JIT kernel 尚无 AMD CI 覆盖,回退到未过滤的 AOT `sgl_kernel` 版本——由于下游 fused MoE down kernel 对被过滤 expert 写零且在读 activation 输入前就 return,写真实输出到这些行是无害的。

## 关键改动

```text
python/sglang/jit_kernel/csrc/elementwise/activation.cuh: kFilterExpert 模板参数 + run_activation_filtered 入口
python/sglang/jit_kernel/activation.py: silu/gelu_and_mul 系列增加 expert_ids/expert_step 参数
python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py: filter_expert 路径改调带过滤的 JIT kernel
python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py: 删除 act_and_mul_triton/_apply_activation/tanh
python/sglang/jit_kernel/tests/test_activation.py: 新增 filter_expert 相关测试
```

kernel 中的过滤逻辑:

```cpp
if (token_id >= params.num_tokens) return;
if constexpr (kFilterExpert) {
  if (params.expert_ids[token_id / params.expert_step] == -1) return;
}
```

## 性能声明

PR 正文给出 `bench_activation.py::benchmark_filter`(bf16)的对比:

| op | dim | bs | skip | Unfiltered (μs) | Filtered (μs) |
|---|---|---|---|---|---|
| silu | 8192 | 1024 | 0.50 | 30.26 | 16.13 |
| silu | 4096 | 16384 | 0.50 | 213.83 | 110.20 |
| gelu | 8192 | 1024 | 0.50 | 29.96 | 16.63 |
| gelu | 4096 | 16384 | 0.50 | 214.03 | 116.92 |

- skip_ratio=0 时过滤开销有界(绝对值 ≤ 约 0.3μs,大形状下可忽略,213.89 → 213.63μs)。
- 与被删除的 Triton `act_and_mul_triton` 相比(删除前用本地副本测量),JIT 路径在 launch-bound 场景(小/中 batch,即 decode 时 MoE expert tile 所在场景)快约 2-3x,HBM 带宽受限形状下 ±5% 以内。
- 精度:`pytest python/sglang/jit_kernel/tests/test_activation.py` 348 passed in 23s(135 个原有 + 213 个新增 filter_expert)。

## 验证与回归风险

新增测试覆盖 op x dtype x shape x expert_step ∈ {1, 16}(per-token 与 sorted/TMA 路由),并有两个边界用例:`all_skipped`(全部 -1 时输出必须完全未被触碰,用 NaN sentinel 校验)与 `none_skipped`(无 -1 时与未过滤路径 bit-exact 相等,atol=0, rtol=0),证明过滤机制不扰动数值。新增 benchmark_filter 对比过滤与非过滤路径。风险点:HIP/XPU 的 filter_expert 行为变化——不再跳行而是全量计算,依赖下游 down kernel "对过滤行写零、读输入前 return" 的实现细节(`fused_moe_triton_kernels.py:192-208`),若该假设被后续改动破坏会产生数值问题;删除的 `_apply_activation`、`tanh` 经 grep 确认无其他调用方,但依赖它们的下游代码需同步。

## 关联

同区域优化:MoE Triton fused_moe kernel 系列(jit-kernel label 的其他激活/MoE kernel PR)。PR 正文未给出具体前序/后续 PR 编号,仅提到"后续 PR 注册 test_activation.py 到 AMD CI"的计划。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
