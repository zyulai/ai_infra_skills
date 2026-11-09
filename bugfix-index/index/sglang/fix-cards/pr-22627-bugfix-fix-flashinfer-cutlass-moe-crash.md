# PR #22627: [Bugfix] Fix flashinfer_cutlass MoE crash when intermediate_size_per_partition is not 16-aligned

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22627](https://github.com/sgl-project/sglang/pull/22627) |
| 作者 | vguduruTT |
| 合并日期 | 2026-05-26 |
| labels | documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell, npu, run-ci, deterministic, diffusion, model-gateway, mthreads, jit-kernel |
| 热度 | 👍 1 · 💬 5 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论

flashinfer_cutlass MoE 在 per-partition intermediate size 非 16 对齐时启动崩溃;加载后对权重补零对齐并收敛后端解析逻辑。

## 问题现象

`flashinfer_cutlass` MoE 后端在 `intermediate_size_per_partition` 不是 16 的倍数时,服务启动即硬崩溃;典型如 NemotronH FP8(Nemotron Nano 30B A3B):`moe_intermediate_size=7688`,TP=8 时 7688/8=961,非 16 对齐。flashinfer 0.6.3+ 上崩溃;旧版 flashinfer 上则会静默回退到更慢的 kernel 路径。该 bug 是在排查 #21307 报告的 Nemotron TP=8 性能问题时发现的。

## 根因

正文披露:flashinfer_cutlass MoE kernel 要求 `intermediate_size_per_partition` 必须是 16 的倍数,而上述模型在 TP=8 切分后不满足该约束,导致 kernel 崩溃;旧版本则静默走慢路径,性能劣化。

## 修复方案

- `python/sglang/srt/layers/quantization/modelopt_quant.py`(ModelOptFp8MoEMethod):`process_weights_after_loading` 中当 `isp % 16 != 0` 时对 `w13_weight`/`w2_weight` 用 `F.pad` 补零对齐;门控激活(swiglu)下 w13 是 [Up, Gate] 沿 dim1 拼接,需拆开两半分别 pad 再 re-concat,保证 kernel 的对半切分对齐。与 `ModelOptNvFp4FusedMoEMethod`(trtllm 128 对齐)的 post-load padding 模式一致。
- `create_moe_runner` 内解析 AUTO→FLASHINFER_CUTLASS 存入 `self._resolved_backend`,用户 `--moe-runner-backend` 覆盖仍生效;`apply` 用 `_resolved_backend` 分发。
- `server_args.py`:删除 NemotronH 特定的 `moe_runner_backend = "flashinfer_cutlass"` 覆盖。
- `layers/moe/fused_moe_triton/layer.py`:移除 `use_flashinfer_cutlass_padded` 预分配 padding(改为加载后 padding)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:ModelOpt FP8 MoE + flashinfer_cutlass 路径及 moe runner 后端解析;正文实测 NVIDIA-Nemotron-3-Nano-30B-A3B-FP8、TP=8、8xH100:输出 140.14→207.18 tok/s(+47.8%),ITL 7.14→4.83ms(-32.4%);称对精度无影响(仅补零),lm_eval gsm8k 与 baseline 一致;TP=1(无需对齐)已验证无回归。仅 `isp % 16 != 0` 时激活,其余模型路径不变。

## 关联

- https://github.com/sgl-project/sglang/issues/21307

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
