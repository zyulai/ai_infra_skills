# PR #25895: [Bugfix] Fix accuracy issue of TRTLLM FP8 MOE and improve logging

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25895](https://github.com/vllm-project/vllm/pull/25895) |
| 作者 | pavanimajety |
| 合并日期 | 2025-09-30 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
TRTLLM(Flashinfer)DSR1 latency FP8 MOE 内核存在精度问题(由 #23991、#23640 引入),routing bias 未转 dtype 即传入内核导致精度受损(由 diff 推断);修复传参并修正内核选择/日志,使 Deep Gemm 在 Flashinfer MOE 开启时自动禁用。

## 问题现象
使用 TRTLLM DSR1 latency 内核(FP8 MOE)时出现精度问题,对应 issue #25189。复现/验证条件:Blackwell B200 上以 `VLLM_USE_FLASHINFER_MOE_FP8=1`、`VLLM_FLASHINFER_MOE_BACKEND="latency"` 跑 LM Eval,DeepSeek-R1-0528 FP8、tensor_parallel_size=8、gsm8k 5-shot、limit 1319、temperature=0.0、max_gen_toks=32768。修复后结果:flexible-extract exact_match 0.9666±0.0049,strict-match 0.9651±0.0051。另外,启用 Flashinfer MOE 时日志会错误打印内核选择信息(如 DeepGemm 与 FlashInfer 并存),本次一并修正。

## 根因
PR 正文披露:精度问题出自 TRTLLM DSR1 latency kernels,bug 由 #23991 与 #23640 引入(修复 #25189)。正文未进一步描述具体错误机制。从 diff 可描述改动所暗示的机制(由 diff 推断):调用 `flashinfer_fused_moe_blockscale_fp8` 时,`e_score_correction_bias`(routing bias)未先转换成与输入 `x` 一致的 dtype 就传入内核,可能因 dtype 不匹配影响路由打分精度;本次在调用前按需执行 `e_score_correction_bias.to(x.dtype)`。此外原实现把内核结果赋给 `result` 后继续走后续路径,本次直接 `return` 内核结果。

## 修复方案
改了三处:

1. `vllm/model_executor/layers/quantization/fp8.py` `Fp8MoEMethod.apply`:`flashinfer_fused_moe_blockscale_fp8` 调用前将 `e_score_correction_bias` 转为 `x.dtype`(可为 None 时保持 None),并直接返回内核结果;
2. `fp8.py` `__init__`:Flashinfer MOE 后端检测收紧为 CUDA + SM100(Blackwell)+ 环境变量开启时才启用;`allow_deep_gemm` 与 `allow_cutlass_block_scaled_grouped_gemm` 均在 Flashinfer MOE 启用时跳过,并输出 `DeepGemm disabled: FlashInfer MOE is enabled.` 等准确日志;
3. `vllm/utils/deep_gemm.py`:`is_deep_gemm_supported()` 增加 `and not envs.VLLM_USE_FLASHINFER_MOE_FP8`;`is_deep_gemm_e8m0_used()` 在 `VLLM_USE_FLASHINFER_MOE_FP8` 开启时禁用 E8M0 并记日志。

关键 diff 摘录:

```diff
-                result = torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
+                e_score_correction_bias = (e_score_correction_bias.to(
+                    x.dtype) if e_score_correction_bias is not None else None)
+                return torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
                     routing_logits=router_logits.to(torch.float32),
                     routing_bias=e_score_correction_bias,
```

## 影响与回归风险
- 影响窗口:bug 由 #23991 与 #23640 引入(正文披露),即这两个 PR 合并后、本 PR 之前的窗口受影响;具体版本号正文未给出。
- 回归面:影响使用 Flashinfer/TRTLLM FP8 MOE 的 Blackwell(SM100)路径——routing bias 现在会做 dtype 转换(此前直接透传);Flashinfer MOE 启用时 DeepGemm 与 Cutlass block-scaled grouped gemm 均被禁用,改变了 SM100 上内核选择优先级;`is_deep_gemm_supported`/`is_deep_gemm_e8m0_used` 的返回值在 `VLLM_USE_FLASHINFER_MOE_FP8=1` 时变为 False,会影响依赖这两个函数的 FP8 linear/E8M0 判断路径。日志文案变化不影响行为。

## 关联
- #25189(修复目标): https://github.com/vllm-project/vllm/issues/25189
- #23991(引入 bug 的 PR): https://github.com/vllm-project/vllm/pull/23991
- #23640(引入 bug 的 PR): https://github.com/vllm-project/vllm/pull/23640

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
