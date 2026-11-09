# PR #38039: fix(mamba): unify causal_conv1d col* dtype to x (MiniCPM-V-4.6 GDN prefill bf16/fp16 mismatch)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38039](https://github.com/sgl-project/sglang/pull/38039) |
| 作者 | BBuf |
| 合并日期 | 2026-09-05 |
| labels | bug, jit-kernel, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
MiniCPM-V-4.6 GDN prefill 因 fp16 输入混 bf16 conv-state 缓存致 Triton 编译断言失败;把 col* 统一 cast 到 x dtype 修复。

## 问题现象
MiniCPM-V-4.6 无法在 SGLang 上服务,GDN prefill 的 conv1d kernel 编译期报 Triton 断言错误:

```
AssertionError: Mismatched type for col0 between then block (<['256'], bf16>) and else block (<['256'], fp16>)
```

触发条件是该模型混合了 fp16 的 `x` 流与 bf16 的 conv-state 缓存(见根因)。影响面:该模型完全起不来;正文指出 vLLM 0.28 因无关的 `k_proj` vs `qkv_proj` 权重映射 bug 也无法加载此模型,所以 SGLang 是唯一可用的引擎。

## 根因
正文披露:MiniCPM-V-4.6 的 LLM 部分是 Qwen3.5(conv-state 缓存为 bf16),但其 vision resampler 输出 **fp16** 激活,于是 GDN prefill 中 fp16 的 `x` 流与 bf16 的 conv-state 缓存混用。`col0..col3` 滑窗寄存器在整个 kernel 内 dtype 不一致——`chunk_offset==0` 分支从 bf16 conv-state 缓存加载它们,而 `chunk_offset>0` 分支以及逐 token 的滑窗重新赋值产生 fp16——Triton 的 if/else phi 类型检查因此中止。

## 修复方案
改 `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py` 的 `_causal_conv1d_fwd_kernel`:`HAS_INITIAL_STATES` 分支里所有从 conv_states 加载的 `col0..col3` 统一 cast 到 `x_ptr.dtype.element_ty`,并保持零初始化也在同一 x dtype 上,使 `col*` 在整个 kernel 内只有单一 dtype。dtype 已一致时(常见情况)cast 是零开销的 no-op。关键改动:

```diff
+            x_elem_ty = x_ptr.dtype.element_ty
             prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
             mask_w = idx_feats < dim
             if KERNEL_WIDTH == 2:
                 conv_states_ptrs = prior_tokens  # [BLOCK_N]
-                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
+                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
```

(KERNEL_WIDTH == 3/4/5 的 col0..col3 各加载点同样加 `.to(x_elem_ty)`。)

另在 `test/registered/layers/mamba/test_causal_conv1d.py` 新增 `test_causal_conv1d_varlen_mixed_input_and_state_dtype`:fp16 输入 + bf16 初始状态(含 conv-state 回写)与参考实现对比。

## 影响与回归风险
- 回归面:所有走 `_causal_conv1d_fwd_kernel`(continuous batching)的模型/路径。dtype 已一致时行为不变(cast 为 no-op);dtype 不一致时此前直接编译失败,不存在静默行为回退。conv-state 回写 dtype 语义由新增 UT 覆盖。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
