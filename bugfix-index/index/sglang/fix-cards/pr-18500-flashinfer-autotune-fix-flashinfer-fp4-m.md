# PR #18500: [Flashinfer Autotune] Fix FlashInfer FP4 MoE autotuning crash by removing incorrect flatten on hidden_states_scale

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18500](https://github.com/sgl-project/sglang/pull/18500) |
| 作者 | YAMY1234 |
| 合并日期 | 2026-02-12 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 13 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
FP4 MoE 路径对 `hidden_states_scale` 多余 `.flatten()` 破坏 2D 形状,`flashinfer_trtllm` 后端 autotune 阶段断言崩溃;删掉 flatten 修复。

## 问题现象
使用 `flashinfer_trtllm` MoE 后端 + FP4 量化(`modelopt_fp4`)时,FlashInfer autotuner 在 `do_preparation` 阶段崩溃,报 `AssertionError: hidden_states_scale's first dimension must be batch size`。调用栈经 `model_runner.py` `_flashinfer_autotune`/`_dummy_run` → `deepseek_v2.py` `forward_normal` → `fused_moe_triton/layer.py` `forward_impl` 中 `trtllm_fp4_block_scale_moe` → flashinfer `fused_moe/core.py` 的断言。发生在 `_dummy_run(batch_size=self.req_to_token_pool.size)` 的 warmup 路径上。

## 根因
PR 正文披露:`hidden_states_scale` 在传入 `trtllm_fp4_block_scale_moe` 前被错误地 flatten 成 1D 张量,违反 FlashInfer API。`_quantize_hidden_states_fp4` 返回的 `hs_scale_linear` 本已是正确的 2D 形状 `[seq_len, hidden_size // 16]`(dtype `float8_e4m3fn`),`.view(torch.float8_e4m3fn)` 是 no-op,而 `.flatten()` 把它错误地转成 1D。这同时违反三处约定:FlashInfer API 要求形状 `[seq_len, hidden_size // 16]`、FlashInfer 中 `hidden_states_scale.dim() == 2` 的显式断言、以及 SGLang 自己在 `_quantize_hidden_states_fp4` 的注释「TRT-LLM expects hidden state scales shaped as [seq_len, hidden_size // 16]」。

## 修复方案
删除 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` `FlashInferFP4MoE.forward_impl` 中 `hidden_states_scale` 上多余的 `.flatten()` 调用:

```diff
-            hidden_states_scale=hs_scale_linear.view(torch.float8_e4m3fn).flatten(),
+            hidden_states_scale=hs_scale_linear.view(torch.float8_e4m3fn),
```

## 影响与回归风险
- 影响窗口:正文未说明;崩溃发生在 autotune 的 `do_preparation`/dummy run 阶段,即所有启用 `flashinfer_trtllm` + `modelopt_fp4` 组合的启动场景。
- 回归面:改动为单行删除,仅影响 `trtllm_fp4_block_scale_moe` 的 `hidden_states_scale` 入参形状。修复后传入的 2D 张量本就是上游 `_quantize_hidden_states_fp4` 的原生输出,与 FlashInfer API、断言及 SGLang 注释三方约定一致;其他 MoE 后端路径不受影响。若下游有代码依赖被 flatten 的 1D 返回形状(无正文证据),需另行确认。

## 关联
- 修复 issue:sgl-project/sglang#18499

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
