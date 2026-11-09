# PR #28073: fix: Fix DSR1 perf regression due to unnecessarily falling back to triton gemm

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28073](https://github.com/sgl-project/sglang/pull/28073) |
| 作者 | trevor-m |
| 合并日期 | 2026-06-15 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
#22300 使 DeepSeek-R1 fp8 gemm 不必要地回退 triton、吞吐从 652 降至 510 tok/s;回退条件改按 dtype 判断恢复性能。

## 问题现象
DeepSeek-R1-0528 出现性能回归。正文基准(40 prompts、input/output 各 1024 tokens、`--fp8-gemm-backend flashinfer_trtllm`):PR #22300 之前(好性能,commit 2c3e84af)输出吞吐 652.59 tok/s、Mean TTFT 205.10 ms;#22300 之后(回归,commit 5da265de)输出吞吐 510.47 tok/s、Mean TTFT 528.37 ms。触发条件是 #22300 让 fp8 gemm 不必要地回退到 triton。

## 根因
正文披露:#22300 为修 MiniMax-M2 的性能与精度问题引入的改动,意外在 DeepSeek-R1-0528 上造成回归——fp8 gemm 被不必要地回退到 triton。机制(正文 Modifications 与 diff 中新增单测的说明一致):`--fp8-gemm-backend flashinfer_trtllm` 下,dense block-FP8 权重 scale 是普通 float32(只有 DeepGEMM 分发路径才 requant 到 UE8M0),TRTLLM groupwise GEMM 本可直接消费 float32 scale;旧回退条件却要求 `weight_scale.format_ue8m0` 为真,导致每个此类 bf16 层都被错误地赶去走慢的 triton 路径。而 #22300 对 MiniMax-M2.5 的真正意义是:TRTLLM GEMM 仅对 bf16 输出数值正确,fp16 输出才需回退——回退判断应落在输出 dtype(与 K>=256)上,而非 scale 的 `format_ue8m0` 属性。

## 修复方案
- `python/sglang/srt/layers/quantization/fp8_utils.py` 的 `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`:回退条件由 `input_2d.shape[1] < 256 or not getattr(weight_scale, "format_ue8m0", False)` 改为 `input_2d.shape[1] < 256 or input_2d.dtype != torch.bfloat16`,与 #22300 在权重加载时加的检查口径一致。
- `python/sglang/srt/model_loader/utils.py`:删除 #22300 rebase 期间引入的死代码 `post_load_weights`。
- 新增单测 `test/registered/unit/layers/quantization/test_flashinfer_trtllm_fp8_fallback.py`,固定"bf16+fp32 scale 走 trtllm、fp16 与 K<256 回退 triton"的行为(mock 后端,可跑 CPU CI)。

关键 diff:
```diff
-    if backend == "trtllm" and (
-        input_2d.shape[1] < 256 or not getattr(weight_scale, "format_ue8m0", False)
+    if backend == "trtllm" and (
+        input_2d.shape[1] < 256 or input_2d.dtype != torch.bfloat16
     ):
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本区间,即 #22300 合入后至本修复之间(基准对比给了回归 commit 5da265de 与修复前好性能 commit 2c3e84af)。
- 回归面:使用 `flashinfer_trtllm` groupwise block-FP8 GEMM 的模型路径——bf16 输出 + fp32 scale 的层恢复走 TRTLLM kernel(性能恢复);fp16 输出与 K<256 仍回退 triton,保留 #22300 的精度保护。修复后基准:输出吞吐 650.43 tok/s、Mean TTFT 205.93 ms,与回归前持平。MiniMax-M2 的 fp16 回退路径与新增单测行为需回归验证。

## 关联
- [#22300](https://github.com/sgl-project/sglang/pull/22300)(正文指出的回归来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
