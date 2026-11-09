# PR #45935: [Model]Fix MiniMaxM2ForCausalLM perf  regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45935](https://github.com/vllm-project/vllm/pull/45935) |
| 作者 | jeejeelee |
| 合并日期 | 2026-06-21 |
| labels | ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
torch.compile 无法融合 QK RMSNorm 的 glue 算子致 MiniMax-M2.5 性能回归;改为手写 Triton 内核手动融合。

## 问题现象
nvidia/MiniMax-M2.5-NVFP4(TP4 on GB200*4)的 PERF benchmark 出现性能回归(正文附基准截图)。修复后精度评估:GPQA Diamond 0.833 / AIME 2025 0.858,官方分数为 0.839 / 0.853。正文未描述具体报错或复现日志。

## 根因
正文明确披露:"The root cause is that torch.compile can't fuse these torch glue ops, which leads to the M25 perf regression." 即 torch.compile 无法融合这些 torch glue 算子,导致 M25 性能回归。

## 修复方案
改 `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`:把原先用 `torch.compile` 包裹的 `_minimax_qk_norm_fallback` 拆成围绕 TP all-reduce 的两个手写 Triton 内核——`_minimax_qk_var_kernel`(all-reduce 前,按 token 对 q/k 段以 fp32 累加均方值,不物化 fp32 的 q/k 副本)与 `_minimax_rms_apply_kernel`(all-reduce 后做 `x * rsqrt(var / tp_world + eps) * weight`,并把 `/tp_world` 折进 rsqrt),由 `_minimax_qk_norm_tp_fallback` 编排;无 Triton 时退回纯 torch 参考路径 `_minimax_qk_norm_tp_eager`。相比 torch.compile 路径避免物化 fp32 q/k 副本和 cat/chunk 临时张量。`forward_qkv` 的 tp_size=1 分支直接走本地路径。同步更新 `tests/kernels/core/test_minimax_reduce_rms.py`(新增 Triton fallback 单卡测试)。

```diff
-@torch.compile(backend=current_platform.simple_compile_backend, dynamic=True)
-def _minimax_qk_norm_fallback(
+@triton.jit
+def _minimax_qk_var_kernel(
+    qkv_ptr,  # [num_tokens, hidden], 16-bit activations
+    var_ptr,  # [num_tokens, 2], fp32
+    row_stride,  # element stride between tokens in qkv
+    q_size: tl.constexpr,  # constant per deployment -> loops unroll, mask elides
+    kv_size: tl.constexpr,
+    BLOCK: tl.constexpr,
+):
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:按 diff 实际改动,MiniMax QK RMSNorm TP 路径数值语义有调整——`forward_qk` 不再做方差 all-reduce(成为 tp==1/已归约构件),多卡参考改走 `_minimax_qk_norm_tp_eager`;Triton 不可用时回退 eager;测试改用 monkeypatch 掉 all-reduce 的单卡校验。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
