# PR #38423: [NVIDIA] Bugfix NVFP4 DGX Spark and RTX50

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38423](https://github.com/vllm-project/vllm/pull/38423) |
| 作者 | johnnynunez |
| 合并日期 | 2026-03-30 |
| labels | bug, ci/build, nvidia, ready, ready-run-all-tests |
| 热度 | 👍 5 · 💬 18 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
SM12x(RTX 50 SM120 / DGX Spark SM121)跑 NVFP4 模型报 cudaErrorIllegalInstruction;升级 CUTLASS/FlashInfer 并补齐按 SM 的编译与运行时守卫。

## 问题现象
在 SM12x GPU(RTX 50 系列 SM120、DGX Spark SM121)上运行 NVFP4 模型(如 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`)时出现 `cudaErrorIllegalInstruction`。不同子场景另有 FlashInfer CUTLASS MoE 报 `Failed to initialize cutlass TMA WS grouped gemm`;只编译 SM100 SASS 时,错误以异步形式在 `synchronize()` 处表现为不透明的 CUDA error。构建含 `ENABLE_NVFP4_SM120` 时修复后可选原生 CUTLASS;不含时正确回退 Marlin(Marlin 在 SM121 稳定,~558 tok/s,与 vLLM CUTLASS ~562 tok/s 相当)。

## 根因
正文披露五点:
1. 捆绑的 CUTLASS 缺少 SM120f family 级 NVFP4/MX Grouped GEMM 编译支持与 SM121 专用 tile 配置,decode 选到 small-M tile 变体时触发 IllegalInstruction(关联上游 NVIDIA/cutlass#3038)。
2. FlashInfer 0.6.6 捆绑 CUTLASS 4.2.1,SM12x 上 CUTLASS MoE 后端 TMA WS grouped gemm 初始化失败(上游 flashinfer#2798 修复)。
3. `cutlass_scaled_mm_supports_fp4()` 只检查 CUDA runtime 版本(>= 12080),不检查该 SM 的 kernel 是否真的编译进去;只编 `ENABLE_NVFP4_SM100` 时误报 SM12x 可用,随后 dispatch 失败。
4. 量化 kernel 无 SM 运行时守卫:只要编译了任意 SM1xx 就派发 `_sm1xxa` kernel;只有 SM100 SASS 时 CUDA 会把 SM100 PTX JIT 到 SM120(不同主架构),异步产生非法指令。
5. `select_nvfp4_linear_backend()` 仅凭 `has_device_capability(100)` 选 FlashInfer CUTLASS,未验证其依赖的 vLLM 量化 kernel 是否为当前 SM 编译。

## 修复方案
- `CMakeLists.txt`:CUTLASS 升至 v4.4.2(diff 显示自 v4.2.1,正文表格写 v4.2.2→v4.4.2),启用 SM120f family 编译。
- `docker/Dockerfile`、`Dockerfile.nightly_torch`、`docker/versions.json`、`requirements/cuda.txt`:FlashInfer 0.6.6 → 0.6.7(含 CUTLASS 4.4.2,修复 SM12x TMA grouped GEMM);留 TODO 待 0.6.8(flashinfer#2738 的原生 SM12x NVFP4/MXFP4 group GEMM)。
- `csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu`:`cutlass_scaled_mm_supports_fp4()` 按 `ENABLE_NVFP4_SM100`/`ENABLE_NVFP4_SM120` 编译宏分 SM 区间报告支持。
- `csrc/quantization/fp4/nvfp4_quant_entry.cu`:新增 `nvfp4_quant_sm_supported()`,给 `scaled_fp4_quant`、`scaled_fp4_experts_quant`、`silu_and_mul_nvfp4_quant`、`silu_and_mul_scaled_fp4_experts_quant` 四个入口加运行时 TORCH_CHECK。
- `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`:`select_nvfp4_linear_backend()` 的 FlashInfer CUTLASS 自动选择加 `cutlass_fp4_supported()` 门控,并为所有 FlashInfer 后端加 assert。
- 附带:`machete_mainloop.cuh` DispatchPolicy 补 `ArchTag = arch::Sm90`;trtllm fp8/nvfp4 MoE 的 routing bias 转 bfloat16(flashinfer#2909);`flashinfer_cutlass_moe.py` 非 EP 分支 `quant_scales` 由 `None` 改 `[]`;`select_unquantized_moe_backend` 移除 `use_ep` 参数(FlashInfer CUTLASS 可用性不再要求 EP)。

```diff
-  return cuda_device_capability >= 100 && runtimeVersion >= 12080;
+  if (runtimeVersion < 12080) return false;
+#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
+  if (cuda_device_capability >= 100 && cuda_device_capability < 120)
+    return true;
+#endif
+#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
+  if (cuda_device_capability >= 120 && cuda_device_capability < 130)
+    return true;
+#endif
+  return false;
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:NVFP4 线性与 MoE 后端选择链、量化 kernel 入口、依赖构建版本的 CI/Docker;Marlin 路径完全绕过 vLLM 量化 kernel,不受新守卫影响、仍是合法回退;SM100(B200)、SM89/SM90 在测试计划中验证无回归;CUTLASS 与 FlashInfer 版本 bump 属全局构建变更,影响所有 CUDA 构建;`select_unquantized_moe_backend` 签名变化与 EP 判定删除会改变未量化 MoE 的后端选择行为。

## 关联
- [NVIDIA/cutlass#3038](https://github.com/NVIDIA/cutlass/pull/3038)
- [flashinfer-ai/flashinfer#2798](https://github.com/flashinfer-ai/flashinfer/pull/2798)
- [flashinfer-ai/flashinfer#2738](https://github.com/flashinfer-ai/flashinfer/pull/2738)
- [flashinfer-ai/flashinfer#2909](https://github.com/flashinfer-ai/flashinfer/issues/2909)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
