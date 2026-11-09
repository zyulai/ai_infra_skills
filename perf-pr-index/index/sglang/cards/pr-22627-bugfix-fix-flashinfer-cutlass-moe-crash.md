# PR #22627: [Bugfix] Fix flashinfer_cutlass MoE crash when intermediate_size_per_partition is not 16-aligned

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vguduruTT |
| 合并 | 2026-05-26 |
| 分类 | MoE(tier: major) |
| Label | documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell, npu, run-ci, deterministic, diffusion, model-gateway, mthreads, jit-kernel |
| Diff | diffstat 缺失(pack 未记录);pack 只保留了部分 diff(modelopt_quant.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/22627 |

## 一句话结论

修复 `intermediate_size_per_partition` 非 16 对齐时 flashinfer_cutlass MoE 的崩溃,补齐对齐 padding 后 NemotronH FP8 TP=8 吞吐 +47.8%。

## 动机

flashinfer_cutlass MoE kernel 要求 `intermediate_size_per_partition` 是 16 的倍数。NemotronH FP8(Nemotron Nano 30B A3B)的 `moe_intermediate_size = 7688`,TP=8 时为 7688/8 = 961,非 16 对齐,在 flashinfer 0.6.3+ 上 server 启动直接硬崩溃;在更老的 flashinfer 版本上则静默回退到更慢的 kernel 路径。该 bug 是作者在排查 #21307 报告的 Nemotron TP=8 性能特征时发现的——也就是说这条模型此前根本没享受到 flashinfer_cutlass 的性能。

## 优化点

修法是权重后处理 padding,遵循 `ModelOptNvFp4FusedMoEMethod` 为 flashinfer_trtllm 做 128-alignment 的同一模式。在 `ModelOptFp8MoEMethod.process_weights_after_loading` 中:当 `intermediate_size_per_partition % 16 != 0` 时,对 `w13_weight` 与 `w2_weight` 用 `F.pad` 零填充到 16 对齐。关键细节是 gated 激活(swiglu)下 w13 是 [Up, Gate] 沿 dim 1 拼接的,必须先拆成两半、各自 pad 再重新 concat,否则 kernel 的半拆分会对不齐。同时把 moe_runner_backend 的 auto-resolution(AUTO -> FLASHINFER_CUTLASS)从 `server_args.py` 移入 `create_moe_runner` 存为 `self._resolved_backend`(沿用 fp8.py 模式,用户 `--moe-runner-backend` 覆盖仍然生效;只有 TRITON 才实例化 MoeRunner),`process_weights_after_loading`/`apply` 改用 resolved backend 而非全局 `get_moe_runner_backend()`,并删除 `server_args.py` 里 NemotronH 专属的 flashinfer_cutlass override 和 `layer.py` 里 load 前预分配 padding 的旧 flag。

## 关键改动

```text
python/sglang/srt/layers/quantization/modelopt_quant.py: post-load 16 字节对齐 padding(gated 时拆 Up/Gate 分别 pad 再 concat),AUTO->FLASHINFER_CUTLASS 解析移入 create_moe_runner
python/sglang/srt/server_args.py: 删除 NemotronH 专属 moe_runner_backend override
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: 移除 use_flashinfer_cutlass_padded 预分配 padding 旧路径
```

gated 权重 padding 关键逻辑:

```python
num_shards = 2 if layer.moe_runner_config.is_gated else 1
isp = layer.w13_weight.shape[1] // num_shards
if isp % 16 != 0:
    pad_amount = round_up(isp, 16) - isp
    # split w13 into [Up, Gate], pad each half separately, re-concat
```

## 性能声明

条件:NVIDIA-Nemotron-3-Nano-30B-A3B-FP8,TP=8,8xH100。Before(Triton MoE 回退)-> After(flashinfer_cutlass + fix):output tok/s 140.14 -> 207.18(+47.8%);ITL 7.14 ms -> 4.83 ms(-32.4%)。TP=8 时 961 -> 976(padding 后)。TP=1 无需对齐(7688 可被 16 整除),已验证无回归。正文未注明具体的请求 workload/并发设置。

## 验证与回归风险

精度:lm_eval gsm8k 与 baseline 一致(正文称 fix 只用零填充权重,无精度影响,与 flashinfer_trtllm 128-alignment 相同做法);另附 TP8_BENCHMARKING_RESULTS.md 与 ACC_VALIDATION.md 链接。CI 状态:Latest PR Test (Base) 进行中,Latest PR Test (Extra) 失败(:x:)。风险面:padding 仅在 `intermediate_size_per_partition % 16 != 0` 时激活,常规对齐模型不受影响;auto-resolution 移入 quant method 后依赖 `self._resolved_backend` 的一致性;旧 flashinfer 版本上静默回退的行为被显式化。

## 关联

相关:#21307(发现本 bug 的性能排查 PR)。同区域:flashinfer_trtllm 的 128-alignment padding(`ModelOptNvFp4FusedMoEMethod`,本 PR 模式的来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
