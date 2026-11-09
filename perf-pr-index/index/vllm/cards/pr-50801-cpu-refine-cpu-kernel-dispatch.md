# PR #50801: [CPU] Refine CPU kernel dispatch

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bigPYJ1151 |
| 合并 | 2026-08-03 |
| 分类 | Quantization(tier: major) |
| Label | documentation, ci/build, cpu |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/50801 |

## 一句话结论

CPU 端把 INT8 W8A8 的 AMX kernel 改为默认启用并修正 MoE 设备/形状门控，oneDNN 升 v3.13，TPOT/吞吐约 +2-3%。

## 动机

源自 SGLang 的 INT8 W8A8 linear AMX kernel 此前被实验开关 `VLLM_CPU_SGL_KERNEL` 挡在后面，正文称 benchmark 显示在其既有 eligibility 窗口内始终开启无副作用；`check_cpu_sgl_kernel` 还错误地排除了 `torch.float16`。审计全部 SGLang GEMM/MoE 选择条件时又发现四个量化专家类的 `_supports_current_device()` 缺少 x86/AMX 检查，在非 AMX x86 或 ARM 上会调用仅编进 AMX 版 `_C` 的 kernel 而抛 `AttributeError`；`CPUExpertsInt8` 还缺 shape 对齐门控。

## 优化点

(1) 去掉 `VLLM_CPU_SGL_KERNEL`：在其资格窗口内无条件启用 AMX SGL kernel，不合格情形（非对称量化、shape 未对齐、非 AMX x86）仍回退 oneDNN，未量化 FP32/FP16/BF16 GEMM 继续只用 oneDNN；(2) `check_cpu_sgl_kernel` 的 dtype 门加入 `torch.float16`，并从 `dispatch_cpu_unquantized_gemm` 删掉已死的 SGL 分支；(3) 给 `CPUExpertsFp8`/`Mxfp4`/`Int4`/`Int8` 的 `_supports_current_device()` 补 x86 + `torch.cpu._is_amx_tile_supported()` 检查，并给 `CPUExpertsInt8` 增加 `is_supported_config` 的 32 对齐门（`hidden_dim` 与 `intermediate_size_per_partition` 均为 32 的倍数）；(4) x86 oneDNN FetchContent pin 由 v3.10 升到 v3.13，ARM pin 不变。

## 关键改动

```text
vllm/envs.py: 移除 VLLM_CPU_SGL_KERNEL 声明与解析
vllm/model_executor/kernels/linear/scaled_mm/cpu.py: INT8 SGL-vs-oneDNN 派发去掉 flag 判断
vllm/model_executor/layers/utils.py: check_cpu_sgl_kernel 纳入 torch.float16，删未量化 GEMM 的 SGL 分支
vllm/model_executor/layers/fused_moe/experts/cpu_moe.py: 四个专家类补 x86+AMX 设备检查，CPUExpertsInt8 补 32 对齐门控
cmake/cpu_extension.cmake: x86 oneDNN pin v3.10 -> v3.13
```

关键片段（设备门控）：

```diff
     def _supports_current_device() -> bool:
-        return current_platform.is_cpu()
+        return (
+            current_platform.is_cpu()
+            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
+            and torch.cpu._is_amx_tile_supported()
+        )
```

## 性能声明

正文给出两组 serving benchmark（AMX x86 host，1024/1024 输入输出）。oneDNN v3.10→v3.13（6 prompts，三个 BF16 模型）：Llama-3.1-8B-Instruct TTFT 2750→2898 ms、TPOT 103.3→100.1 ms、total tok/s 113.2→116.6；gemma-7b 3255→3092 ms、133.3→130.4 ms、87.9→89.9；Qwen3-8B 2812→2806 ms、106.3→102.4 ms、110.0→114.1。正文结论"No regression; TPOT/throughput improve slightly (~2-3%)"，TTFT 在 6-prompt 样本内属噪声。Linear INT8 flag 移除（`RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8`，15 prompts）：Mean TTFT 4136.7→3721.5 ms、Mean TPOT 79.15→74.09 ms、Total tok/s 359.8→385.1。

## 验证与回归风险

rebase 到 post-#50133 的 main 并干净增量构建，651 tests pass、289 skipped、0 failed（含 `test_onednn.py`、CPU 未量化 GEMM 派发、scaled_mm kernel selection、CPU FP8 scaled_mm、W8A8 e2e、全套 CPU MoE 测试）；四个模型离线 smoke 输出连贯，Llama/Qwen3 在 oneDNN 两版间逐字节一致，gemma-7b 四个 prompt 有一个分叉（BF16 舍入序差异，仍连贯）。设备/形状门控只改变本机无法覆盖的非 AMX x86、ARM、非对齐 shape 行为，shape 拒绝路径另经直接单测确认。PR 明确标注 `CPUExpertsFp8`/`Mxfp4`/`Int4` 仍无 shape 对齐门，留作后续。

## 关联

#50133（先于本 PR 合并，删掉未量化 CPU MoE 的 SGL 派发块与 flag 检查，本 PR 因此得以彻底移除 `VLLM_CPU_SGL_KERNEL`）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
